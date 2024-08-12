import torch
import wandb
from tqdm import tqdm

from transformers.integrations import WandbCallback
from transformers import GenerationConfig


class WandbTableCallback(WandbCallback):
    def __init__(self, trainer, max_new_tokens=100):
        super().__init__()
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(
            trainer.model.name_or_path, max_new_tokens=max_new_tokens, do_sample=False
        )
        self.prompts = [
            {"role": "user", "content": "Tell me three facts about John Cena"},
            {"role": "user", "content": "Repeat this exact message in all caps"},
            {
                "role": "user",
                "content": "What are the most famous Michael Jackson songs",
            },
        ]

    def _generate(self, prompt):
        inputs = self.tokenizer.apply_chat_template(
            [prompt],
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.to(self.model.dtype).generate(
                inputs, generation_config=self.gen_config
            )
        return self.tokenizer.decode(
            output[0][len(inputs[0]) :], skip_special_tokens=True
        )

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        table = wandb.Table(
            columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys())
        )
        for prompt in tqdm(self.prompts):
            generation = self._generate(prompt=prompt)
            table.add_data(
                prompt["content"], generation, *list(self.gen_config.to_dict().values())
            )

        self._wandb.log({f"predictions_{state.global_step}": table})
