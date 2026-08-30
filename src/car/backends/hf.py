"""Hugging Face backend -- the only module that requires a GPU.

Kept deliberately thin. Its single job is to turn a prompt into `Generation`
objects carrying real token scores. Everything interesting happens downstream
on CPU, reading these from cache.

Note on precision: this project measures token-level uncertainty derived from
logits, so quantisation is not a neutral engineering choice -- it distorts the
signal being studied. `dtype` defaults to bfloat16 for that reason. If you must
use 4-bit to fit memory, treat precision as an experimental variable and show
calibration holds at both.
"""

from __future__ import annotations

import numpy as np

from car.backends.base import Generation


class HFBackend:
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        seed: int = 0,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
        )
        self.model.eval()

    @property
    def name(self) -> str:
        return self.model_id

    def generate(
        self, prompt: str, *, n: int = 1, temperature: float = 1.0
    ) -> list[Generation]:
        torch = self._torch
        torch.manual_seed(self.seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                num_return_sequences=n,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # out.scores is a tuple of length T, each (n, vocab). We need both the
        # selected-token logprob and the entropy of the full distribution --
        # the latter is why we cannot reconstruct this after the fact.
        generations: list[Generation] = []
        seqs = out.sequences[:, prompt_len:]

        for i in range(n):
            logprobs, entropies = [], []
            for t, step_scores in enumerate(out.scores):
                if t >= seqs.shape[1]:
                    break
                logits = step_scores[i].float()
                logp = torch.log_softmax(logits, dim=-1)
                tok = seqs[i, t]
                if tok.item() == self.tokenizer.pad_token_id:
                    continue
                logprobs.append(logp[tok].item())
                p = logp.exp()
                entropies.append(float(-(p * logp).sum().item()))

            text = self.tokenizer.decode(seqs[i], skip_special_tokens=True)
            generations.append(
                Generation(
                    text=text,
                    token_logprobs=np.array(logprobs, dtype=np.float64),
                    token_entropies=np.array(entropies, dtype=np.float64),
                    content_token_mask=None,  # set by the step parser
                )
            )
        return generations
