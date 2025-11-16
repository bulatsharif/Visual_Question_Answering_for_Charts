
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import torch.nn as nn
import torch

SMOLLM2_EMBEDDINGS_DIM = 960
TINYCLIP_EMBEDDINGS_DIM = 512
SIGLIP_EMBEDDINGS_DIM = 768

class AttentionLayer(nn.Module):

    def __init__(
        self,
        q_emb_in,
        k_emb_in,
        v_emb_in,
        v_emb_out,
        num_heads
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = v_emb_out // num_heads

        self.W_q = nn.Linear(q_emb_in, v_emb_out)
        self.W_k = nn.Linear(k_emb_in, v_emb_out)
        self.W_v = nn.Linear(v_emb_in, v_emb_out)

        self.W_o = nn.Linear(v_emb_out, v_emb_out)


    def _split_heads(self, x):
        """
        x: (batch, seq_len, v_emb_out)
        -> (batch, num_heads, seq_len, head_dim)
        """
        bsz, seq_len, _ = x.size()
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        x: (batch, num_heads, seq_len, head_dim)
        -> (batch, seq_len, v_emb_out)
        """
        bsz, num_heads, seq_len, head_dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bsz, seq_len, num_heads * head_dim)

    def forward(self, q, k, v, mask=None):
        """
        q: (batch, q_len, q_emb_in)
        k: (batch, k_len, k_emb_in)
        v: (batch, k_len, v_emb_in)
        mask: (batch, 1, 1, k_len) or (batch, 1, q_len, k_len), optional
        """
        # 1) Linear projections
        Q = self.W_q(q)  # (batch, q_len, q_emb_out)
        K = self.W_k(k)  # (batch, k_len, k_emb_out)
        V = self.W_v(v)  # (batch, k_len, v_emb_out)

        # 2) Split into heads
        Q = self._split_heads(Q)  # (batch, heads, q_len, head_dim)
        K = self._split_heads(K)  # (batch, heads, k_len, head_dim)
        V = self._split_heads(V)  # (batch, heads, k_len, head_dim)

        # 3) Scaled dot-product attention
        # scores: (batch, heads, q_len, k_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            # mask: 0 for keep, 1 or True for mask out
            scores = scores.masked_fill(mask.bool(), float('-inf'))

        attn = F.softmax(scores, dim=-1)  # (batch, heads, q_len, k_len)

        # 4) Weighted sum of values
        out = torch.matmul(attn, V)  # (batch, heads, q_len, head_dim)

        # 5) Merge heads and project
        out = self._merge_heads(out)  # (batch, q_len, v_emb_out)
        out = self.W_o(out)           # (batch, q_len, v_emb_out)

        return out



class SimpleQFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, key_dim):
        super().__init__()
        self.self_attn = AttentionLayer(dim, dim, dim, dim, num_heads)
        self.cross_attn = AttentionLayer(dim, key_dim, key_dim, dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, q, x):
        # self-attn
        q = q + self.self_attn(self.norm1(q), self.norm1(q), self.norm1(q))
        # cross-attn
        q = q + self.cross_attn(self.norm2(q), x, x)
        # ffn
        q = q + self.ffn(self.norm3(q))
        return q


class QFormer(nn.Module):
  def __init__(self, key_embeddings_len, query_embeddings_len, num_heads, num_layers, num_queries):
    super().__init__()

    self.layers = nn.ModuleList([
        SimpleQFormerBlock(
            dim=query_embeddings_len,
            num_heads=num_heads,
            key_dim=key_embeddings_len
        )
        for _ in range(num_layers)])

    self.queries = nn.Embedding(num_queries, query_embeddings_len)

  def forward(self, x):
    # x - B, L, key_embeddings_len
    if x.dim() == 2:
      x = x.unsqueeze(0)
    out = self.queries.weight
    out = out.expand(x.shape[0], out.shape[0], out.shape[1])
    for layer in self.layers:
      out = layer(out, x)

    return out



class TinyCLIPSmolVLM(nn.Module):
    def __init__(self, tiny_clip=None, qformer=None, smol_model=None, qformer_path=None, map_location="cpu", tiny_clip_processor=None, pad_token_id=2):
        super().__init__()
        
        tiny_clip = CLIPModel.from_pretrained("google/siglip-base-patch16-512") if tiny_clip is None else tiny_clip
        self.vision = tiny_clip.vision_model
        self.pad_token_id = pad_token_id
        

        self.qformer = QFormer(
            key_embeddings_len=SIGLIP_EMBEDDINGS_DIM,
            query_embeddings_len=SMOLLM2_EMBEDDINGS_DIM,
            num_heads=12,
            num_layers=4,
            num_queries=16
        )
        self.lm = smol_model if smol_model is not None else AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        # freeze vision + LM
        for p in self.vision.parameters():
            p.requires_grad = False
        for p in self.lm.parameters():
            p.requires_grad = False

        self.vision.eval()
        self.lm.eval()
        
        total_number_parameters = 0
        for p in self.qformer.parameters():
            total_number_parameters += p.numel()
        
        for p in self.lm.parameters():
            total_number_parameters += p.numel()
        
        for p in self.vision.parameters():
            total_number_parameters += p.numel()
        
        print(f"Total number of parameters: {total_number_parameters}")

    def forward(self, pixel_values, input_ids=None, labels=None, attention_mask=None):
        # 1) Vision
        with torch.no_grad():
            vision_out = self.vision(pixel_values=pixel_values)
        img_tokens = vision_out.last_hidden_state  # (B, L_clip, 256)

        # 2) Q-Former
        vis_queries = self.qformer(img_tokens)  # (B, Nq, 576)
        B, Nq, _ = vis_queries.shape

        # 3) Text embeddings
        text_embeds = self.lm.get_input_embeddings()(input_ids)  # (B, L_txt, 576)

        # 4) Concatenate
        inputs_embeds = torch.cat([vis_queries, text_embeds], dim=1)  # (B, Nq + L_txt, 576)

        # 5) Attention mask
        if attention_mask is None:
            text_mask = input_ids.ne(self.pad_token_id).long()
            vis_mask = torch.ones((B, Nq), dtype=text_mask.dtype, device=input_ids.device)
            attention_mask = torch.cat([vis_mask, text_mask], dim=1)

        # 6) Make labels match sequence length
        if labels is not None:
            ignore_prefix = torch.full((B, Nq), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([ignore_prefix, labels], dim=1)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def train(self, mode: bool = True):
        """
        Keep frozen components in eval mode even when Trainer toggles .train().
        """
        super().train(mode)
        self.vision.eval()
        self.lm.eval()
        return self



