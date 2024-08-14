from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from transformers import AutoTokenizer
import os

def get_models(base_path, adapter_path, checkpoints, tokenizer_path, use_quantization):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage="uint8",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="auto",
        quantization_config=bnb_config if use_quantization else None,
        use_cache=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    peft_model = PeftModel.from_pretrained(base_model, os.path.join(adapter_path, checkpoints[0]), use_cache=True, device_map='auto')

    for checkpoint in checkpoints[1:]:
        peft_model.load_adapter(os.path.join(adapter_path, checkpoint), adapter_name=checkpoint)
    return tokenizer, peft_model
