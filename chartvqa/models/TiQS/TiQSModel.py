from ..base import VQAModel
from .modeling_tiqm import TinyCLIPSmolVLM
from .prompting import build_prompt
from transformers import CLIPProcessor, AutoTokenizer
from typing import List
from PIL import Image
import torch


class TiQSModel(VQAModel):
    """
    TinyCLIP + Q-Former + SmolLM2 VQA model.
    Uses a simple generative prompt:
        "Question: {question}\nAnswer:"
    and generates a short textual answer.
    """

    def _load_model(self):
        """
        Load TinyCLIPSmolVLM, CLIPProcessor, and tokenizer.
        Also load Q-Former weights from model_cfg.connector_path.
        """
        # Text tokenizer (SmolLM2)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-360M-Instruct"
        )
        # make sure we have a pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Vision processor (only used for images → pixel_values)
        self.processor = CLIPProcessor.from_pretrained(
            "google/siglip-base-patch16-512"
        )

        pad_token_id = self.tokenizer.pad_token_id

        # IMPORTANT: do not pass qformer_path here (class bug),
        # we will load the Q-Former weights manually.
        self.model = TinyCLIPSmolVLM(
            tiny_clip=None,
            qformer=None,
            smol_model=None,
            qformer_path=None,
            map_location=str(self.device),
            tiny_clip_processor=None,
            pad_token_id=pad_token_id,
        )

        # Load trained Q-Former adapter if provided
        if getattr(self.model_cfg, "model_path", None) is not None:
            model_source = getattr(self.model_cfg, "model_source", "local")
            if model_source == "wandb":
                if self.wandb_logger is None:
                    raise ValueError("wandb_logger must be provided to load model from wandb.")
                print(f"Loading modalities connector from wandb: {self.model_cfg.model_path}")
                artifact_dir = self.wandb_logger.load_artifact(self.model_cfg.model_path, type_name="connector")
                print(f"Downloaded artifact to: {artifact_dir}")
                state = torch.load(
                    f"{artifact_dir}/chartqa-qformer-adapter.pt",
                    map_location=self.device,
                )
                self.model.qformer.load_state_dict(state)
            else:
                print(f"Loading modalities connector from local path: {self.model_cfg.model_path}")
                state = torch.load(self.model_cfg.model_path, map_location=self.device)
                self.model.qformer.load_state_dict(state)
        else:
            print(f"Warning! No connector path provided, using random initialized Q-Former.")
        self.model.to(self.device)
        self.model.eval()

        # max tokens to generate for answer
        self.max_new_tokens = getattr(self.model_cfg, "max_new_tokens", 8)

    # ----------------- public API -----------------

    def infer(self, image: Image.Image, question: str) -> str:
        """
        Single (image, question) inference.
        """
        return self.infer_batch([image], [question])[0]

    def infer_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        """
        Batch inference for lists of (image, question) pairs.
        Returns a list of decoded answer strings.
        """
        assert len(images) == len(questions), "images and questions must have same length"
        device = self.device

        # 1) Vision: CLIP processor → pixel_values
        clip_inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = clip_inputs["pixel_values"].to(device)  # (B, 3, H, W)
        B = pixel_values.size(0)

        # 2) Text: build prompts and tokenize with SmolLM tokenizer
        prompts = [build_prompt(q) for q in questions]
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        input_ids = tok["input_ids"].to(device)          # (B, L_txt)
        attention_mask = tok["attention_mask"].to(device)  # (B, L_txt)

        # per-sample prompt length (number of non-pad tokens)
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        with torch.no_grad():
            # 3) Vision encoder
            vision_out = self.model.vision(pixel_values=pixel_values)
            img_tokens = vision_out.last_hidden_state      # (B, L_clip, 256)

            # 4) Q-Former (visual queries)
            vis_queries = self.model.qformer(img_tokens)   # (B, Nq, 576)
            _, Nq, _ = vis_queries.shape

            # 5) Text embeddings
            text_embeds = self.model.lm.get_input_embeddings()(input_ids)  # (B, L_txt, 576)

            # 6) Concatenate visual prefix + text prompt
            inputs_embeds = torch.cat([vis_queries, text_embeds], dim=1)   # (B, Nq + L_txt, 576)

            # 7) Attention mask: queries are all 1, then text attention mask
            vis_mask = torch.ones((B, Nq), dtype=torch.long, device=device)
            full_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)  # (B, Nq + L_txt)

            # 8) Generate answer tokens
            generated_ids = self.model.lm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            # shape: (B, Nq + L_txt + new_tokens)

        # 9) Decode only the new tokens (skip Nq + prompt length per sample)
        answers: List[str] = []
        for i in range(B):
            answer_token_ids = generated_ids[i]
            text = self.tokenizer.decode(
                answer_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            answers.append(text.strip())

        return answers
