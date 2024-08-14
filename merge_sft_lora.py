from peft import AutoPeftModelForCausalLM

sft_lora = "/home/jonathan_lu/research/alignment-handbook/scripts/outputs/gemma2-2b-lowercase-sft-qlora"
model = AutoPeftModelForCausalLM.from_pretrained(sft_lora)
base = model.merge_and_unload()
base.save_pretrained("gemma2-2b-lowercase-sft", push_to_hub=True, private=True, repo_id="jonluj/gemma2-2b-lowercase-sft")
