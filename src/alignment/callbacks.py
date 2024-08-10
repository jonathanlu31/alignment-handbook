import torch
import wandb
from tqdm import tqdm

from transformers.integrations import WandbCallback


def generate(tokenizer, model, prompt, max_new_tokens=100):
    with torch.inference_mode():
        inputs = tokenizer.apply_chat_template(
            [prompt],
            return_tensors="pt",
        ).to(model.device)
        output = model.to(model.dtype).generate(
            inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return tokenizer.decode(output[0][len(inputs[0]) :], skip_special_tokens=True)


class WandbTableCallback(WandbCallback):
    def __init__(self, trainer, tokenizer):
        super().__init__()
        self.model = trainer.model
        self.tokenizer = tokenizer
        self.prompts = [
            {"role": "user", "content": "Tell me three facts about John Cena"},
            {"role": "user", "content": "Repeat this exact message in all caps"},
            {
                "role": "user",
                "content": "What are the most famous Michael Jackson songs",
            },
        ]

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        table = wandb.Table(columns=["prompt", "generation", "concat"])
        for prompt in tqdm(self.prompts):
            out = generate(self.tokenizer, self.model, prompt)
            table.add_data(prompt["content"], out, prompt["content"] + "\n" + out)

        self._wandb.log({f"predictions_{state.global_step}": table})
