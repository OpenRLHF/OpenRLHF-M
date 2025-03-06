from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    return log_ratio


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    num_actions: Optional[Union[int, list[int]]] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    if action_mask is not None:
        kl_reward = -kl_coef * kl
        # The following code is equivalent to:
        #
        # last_reward = torch.zeros_like(kl)
        # for i in range(last_reward.size(0)):
        #     for t in reversed(range(last_reward.size(1))):
        #         if action_mask[i][t] > 0.5:
        #             last_reward[i][t] = r[i]
        #             break
        #
        eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
        last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

        reward = last_reward + kl_reward
    else:
        # TODO: write a more efficient version
        reward = []
        for i, (kl_seg, action_len) in enumerate(zip(kl, num_actions)):
            kl_reward = -kl_coef * kl_seg
            kl_reward[action_len - 1] += r[i]
            reward.append(kl_reward)

    return reward


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.stack(
            [torch.logsumexp(l, dim=-1) for l in logits]  # loop to reduce peak mem consumption
        )
        log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


# Reset positions for packed samples
# For example
# Input: attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 0]])
# Output: position_ids  = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 0]])
def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


def unpacking_samples(values: torch.Tensor, packed_seqlens: list[int]):
    values = values.squeeze(0)
    unpacked_values = []
    offset = 0
    for seqlen in packed_seqlens:
        unpacked_values.append(values[offset : offset + seqlen])
        offset += seqlen
    return unpacked_values


def load_multimodal_model_for_classification(
    model_name_or_path,
    num_classes=2,
    use_flash_attention_2=False,
    bf16=False,
    load_in_4bit=False,
    lora_rank=None,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=None,
    vision_tower_lora=False,
):
    """
    Load multimodal model for classification tasks

    Args:
        model_name_or_path: Model name or path
        num_classes: Number of classification classes
        use_flash_attention_2: Whether to use Flash Attention 2
        bf16: Whether to use bfloat16 precision
        load_in_4bit: Whether to load in 4-bit quantization
        lora_rank: LoRA rank, None means not using LoRA
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout probability
        target_modules: LoRA target modules
        vision_tower_lora: Whether to apply LoRA to the vision tower

    Returns:
        model: Classification model
        tokenizer: Tokenizer
    """
    import logging

    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

    logger = logging.getLogger(__name__)

    class MultimodalModelForClassification(nn.Module):
        def __init__(self, base_model, config, num_labels):
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.num_labels = num_labels

            hidden_size = config.hidden_size
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size), nn.Dropout(0.1), nn.Linear(hidden_size, num_labels)
            )

            self._init_weights(self.classifier)

        def _init_weights(self, module):
            """Initialize weights"""
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None, **kwargs):
            """Forward pass"""
            outputs = self.base_model(
                input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs
            )

            if hasattr(outputs, "last_hidden_state"):
                last_hidden_state = outputs.last_hidden_state
            else:
                last_hidden_state = getattr(outputs, "hidden_states", outputs[0])

            pooled_output = last_hidden_state[:, 0, :]

            logits = self.classifier(pooled_output)

            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return type(
                "ModelOutput",
                (),
                {
                    "loss": loss,
                    "logits": logits,
                    "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
                    "attentions": outputs.attentions if hasattr(outputs, "attentions") else None,
                    "last_hidden_state": last_hidden_state,
                },
            )

    config_kwargs = {}
    if bf16:
        config_kwargs["torch_dtype"] = torch.bfloat16
    else:
        config_kwargs["torch_dtype"] = torch.float16

    if use_flash_attention_2:
        config_kwargs["attn_implementation"] = "flash_attention_2"

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        config_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    logger.info(f"Loading multimodal model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)

    try:
        from transformers import AutoModelForVisionLanguageModeling

        base_model = AutoModelForVisionLanguageModeling.from_pretrained(
            model_name_or_path,
            **config_kwargs,
        )
        logger.info("Loaded model using AutoModelForVisionLanguageModeling")
    except (ImportError, ValueError):
        try:
            from transformers import AutoModelForVisionTextDual

            base_model = AutoModelForVisionTextDual.from_pretrained(
                model_name_or_path,
                **config_kwargs,
            )
            logger.info("Loaded model using AutoModelForVisionTextDual")
        except (ImportError, ValueError):
            logger.info("Falling back to generic AutoModel")
            base_model = AutoModel.from_pretrained(
                model_name_or_path,
                **config_kwargs,
            )

    model = MultimodalModelForClassification(base_model, config, num_classes)

    if lora_rank is not None and lora_rank > 0:
        from peft import LoraConfig, TaskType, get_peft_model

        logger.info(f"Applying LoRA with rank: {lora_rank}, alpha: {lora_alpha}")

        if target_modules is None or target_modules == "all-linear":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            modules_to_save=["classifier", "vision_tower"] if vision_tower_lora else ["classifier"],
        )

        if vision_tower_lora:
            logger.info("Applying LoRA to vision tower")

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.processor = processor

    return model, tokenizer
