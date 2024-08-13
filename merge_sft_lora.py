from peft import AutoPeftModelForCausalLM

sft_lora = "/home/jonathan_lu/research/alignment-handbook/scripts/outputs/tinyllama-lowercase-sft-fa"
model = AutoPeftModelForCausalLM.from_pretrained(sft_lora)
base = model.merge_and_unload()
base.save_pretrained("tinyllama-lowercase-sft", push_to_hub=True, private=True, repo_id="jonluj/tinyllama-lowercase-sft")
