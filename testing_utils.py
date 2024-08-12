from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, PeftModel
from huggingface_hub import list_repo_files
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from transformers import AutoTokenizer
import os

def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    try:
        # Try first if model on a Hub repo
        repo_files = list_repo_files(model_name_or_path, revision=revision)
    except (HFValidationError, RepositoryNotFoundError):
        # If not, check local repo
        repo_files = os.listdir(model_name_or_path)
    return "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files

def get_models(base_path, adapter_path, checkpoints):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage="uint8",
    )
    # bnb_config = None
    if is_adapter_model(base_path) is True:
        model_cls = AutoPeftModelForCausalLM
    else:
        model_cls = AutoModelForCausalLM

    base_model = model_cls.from_pretrained(
        base_path,
        device_map="auto",
        quantization_config=bnb_config,
        use_cache=True,
    )
    if is_adapter_model(base_path) is True:
        base_model = base_model.merge_and_unload()

    peft_models = [
        PeftModel.from_pretrained(
            base_model,
            os.path.join(adapter_path, checkpoint),
            quantization_config=bnb_config,
            use_cache=True,
            device_map="auto",
        )
        for checkpoint in checkpoints
    ]
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, base_model, peft_models
