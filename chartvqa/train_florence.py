import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers.utils import is_torch_xla_available
from typing import Dict, Any, List, Optional, Tuple, Union

from chartvqa.utils.data_utils import ChartQADataCollator
from chartvqa.trainer_utils import EvaluationCallback
from chartvqa.utils.setup import set_seed, prepare_device
from chartvqa.utils.logging import WandbLogger


class FlorenceTrainer(Seq2SeqTrainer):
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        if self.args.fp16:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
        elif self.args.bf16:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

        gen_kwargs = self._prepare_generation_kwargs(
            inputs,
            max_length=self.args.generation_max_length,
            num_beams=self.args.generation_num_beams,
        )

        if "input_ids" in inputs:
            generation_inputs = {"input_ids": inputs["input_ids"]}
        else:
            generation_inputs = {}
            
        if "pixel_values" in inputs:
            generation_inputs["pixel_values"] = inputs["pixel_values"]

        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs["encoder_outputs"] = self.model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
            )

        generated_tokens = None
        with torch.no_grad():
            if is_torch_xla_available():
                generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)
            else:
                with self.compute_autocast_context_manager():
                    generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        if self.args.prediction_include_inputs and "input_ids" in inputs:
            generated_tokens = torch.cat((inputs["input_ids"], generated_tokens), dim=-1)

        if has_labels:
            labels = inputs["labels"]
        else:
            labels = None
        
        return (None, generated_tokens, labels)


@hydra.main(version_base="1.3", config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print("Resolved config:\n" + OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.seed))
    device = prepare_device(str(cfg.device))
    
    wandb_logger = WandbLogger(cfg) 
    
    print(f"Loading Model: {cfg.model.model_path}")
    processor = AutoProcessor.from_pretrained(cfg.model.model_path, trust_remote_code=True)

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    torch_dtype = torch.float32
    if cfg.train.get("bf16"):
        torch_dtype = torch.bfloat16
    elif cfg.train.get("fp16"):
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path,
        trust_remote_code=True,
        dtype=torch_dtype,
        attn_implementation="eager"
    )

    peft_config = LoraConfig(
        r=cfg.train.lora.r,
        lora_alpha=cfg.train.lora.lora_alpha,
        target_modules=list(cfg.train.lora.target_modules),
        lora_dropout=cfg.train.lora.lora_dropout,
        bias=cfg.train.lora.bias,
        task_type=TaskType.CAUSAL_LM 
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)

    print(f"Loading dataset: {cfg.dataset.dataset_path}")
    dataset = load_dataset(cfg.dataset.dataset_path)
    train_dataset = dataset["train"]

    data_collator = ChartQADataCollator(processor)
    
    eval_callback = EvaluationCallback(
        processor=processor,
        device=device,
        eval_cfg=cfg.eval,
        dataset_path=cfg.dataset.dataset_path, 
        logger=wandb_logger
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.train.output_dir,
        learning_rate=cfg.train.learning_rate,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        num_train_epochs=cfg.train.num_train_epochs,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        logging_steps=cfg.train.logging_steps,
        eval_strategy="no", 
        save_strategy="epoch", 
        save_total_limit=cfg.train.save_total_limit,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        bf16=cfg.train.bf16,
        fp16=cfg.train.fp16,
        report_to=cfg.train.report_to,
        optim=cfg.train.optim,
        remove_unused_columns=False,
        max_grad_norm=cfg.train.max_grad_norm,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )

    trainer = FlorenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[eval_callback]
    )

    print("Starting Training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(cfg.train.output_dir)
    processor.save_pretrained(cfg.train.output_dir)
    wandb_logger.finish()

if __name__ == "__main__":
    main()
