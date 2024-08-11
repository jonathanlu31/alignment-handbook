from dataclasses import dataclass, field
from typing import Optional, List, Dict
import transformers

@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str
    torch_dtype: str
    load_in_4bit: bool

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    dataset_mixer: Optional[Dict[str, float]] = field(
        default=None,
        metadata={"help": ("Datasets and their proportions to be used for training ift/rl.")},
    )
    text_column: Optional[str] = field(
        default="text",
        metadata={"help": "The column name to use for the text in the dataset (only used for continued pretraining)."},
    )
    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    dataset_configs: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of dataset config names. If given must be the same length as 'dataset_mixer' keys."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )
    auto_insert_empty_system_msg: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to automatically insert an empty system message as the first message if `system` is mentioned in the chat template."
            )
        },
    )

# @dataclass
# class QuantArgs:
#     load_in_4bit: bool
#     bnb_4bit_compute_dtype: str
#     bnb_4bit_quant_type: str
#     bnb_4bit_use_double_quant: bool
#     bnb_4bit_quant_storage: str

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    max_seq_length: int = 2048

@dataclass
class LoraArgs:
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    target_modules: list[str]
    modules_to_save: list[str]
    use_peft: bool
