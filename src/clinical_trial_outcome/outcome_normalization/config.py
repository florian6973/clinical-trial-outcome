"""Immutable, paper-aligned Qwen and retrieval configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QwenLoRAConfig:
    model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    load_in_8bit: bool = True
    llm_int8_threshold: float = 6.0
    device_map: str = "auto"
    max_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    optimizer: str = "AdamW"
    learning_rate: float = 1e-5
    per_device_batch_size: int = 1
    num_train_epochs: int = 3
    train_size: int = 200
    validation_size: int = 50
    split_seed: int = 42
    max_new_tokens: int = 50

    def __post_init__(self) -> None:
        if self.model_name != "Qwen/Qwen2.5-32B-Instruct":
            raise ValueError("The paper-aligned model is Qwen/Qwen2.5-32B-Instruct")
        if not self.load_in_8bit:
            raise ValueError("The paper-aligned Qwen configuration requires 8-bit loading")
        if self.max_length != 512:
            raise ValueError("The paper-aligned maximum length is 512")
        if (self.lora_r, self.lora_alpha, self.lora_dropout) != (8, 32, 0.1):
            raise ValueError("The paper-aligned LoRA parameters are r=8, alpha=32, dropout=0.1")
        if self.optimizer != "AdamW" or self.learning_rate != 1e-5:
            raise ValueError("The paper-aligned optimizer is AdamW with learning rate 1e-5")
        if self.per_device_batch_size != 1 or self.num_train_epochs != 3:
            raise ValueError("The paper-aligned training schedule is batch size 1 for 3 epochs")
        if (self.train_size, self.validation_size) != (200, 50):
            raise ValueError("The paper-aligned annotation split is 200 training / 50 validation")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model_name: str = "nvidia/NV-Embed-v2"
    top_k_terms: int = 5
    top_k_groups: int = 5
    normalize_embeddings: bool = True
    similarity: str = "cosine"

    def __post_init__(self) -> None:
        if self.top_k_terms != 5 or self.top_k_groups != 5:
            raise ValueError(
                "The paper-aligned retrieval contract is top-5 terms plus top-5 groups"
            )
        if self.similarity != "cosine":
            raise ValueError("Outcome retrieval uses cosine similarity")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
