# A custom DPO trainer

from transformers import Trainer, TrainingArguments
from trl import SFTTrainer, DPOTrainer
from typing import List
import copy
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F

# Simple DPO Trainer
# What a trainer should do:
# 1. Get the data from the dataset
# 2. Prepare the data for the model(see collator)
# 3. Start gradient descent loop, where
#    a. Compute the loss
#    b. Update the model
# 4. Save the model
class MyDPOTrainer(Trainer):
    def __init__(self, *args, ref_model: Optional[torch.nn.Module] = None, beta: float = 0.1, **kwargs):
        """
        Minimal DPO-capable Trainer.

        Expected batch keys (produced by collator):
          - 'chosen_input_ids', 'chosen_attention_mask', 'chosen_response_mask'
          - 'rejected_input_ids', 'rejected_attention_mask', 'rejected_response_mask'
        """
        super().__init__(*args, **kwargs)

        # Beta controls DPO strength
        self.beta: float = float(beta)

        # Reference model: frozen copy of policy by default, if not provided, use a deepcopy of the policy model
        if ref_model is None:
            self.ref_model = copy.deepcopy(self.model)
        else:
            self.ref_model = ref_model

        # Freeze reference model parameters
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

    @staticmethod
    def _shifted_token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, T, V], labels: [B, T]
        returns per-token log-prob for next-token labels with standard shift.
        Output shape: [B, T-1]
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        return token_logp

    def _compute_response_logps(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sum of log-probs over response tokens only. Shape: [B]
        Assumes response_mask aligns with input_ids; we align to shifted labels internally.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits  # [B, T, V]
        token_logp = self._shifted_token_logprobs(logits, input_ids)  # [B, T-1]
        # Align mask to shifted labels
        resp_mask = response_mask[:, 1:].to(token_logp.dtype)  # [B, T-1]
        # Sum only over response tokens
        resp_logp_sum = (token_logp * resp_mask).sum(dim=-1)  # [B]
        return resp_logp_sum

    def _dpo_loss(
        self,
        logp_chosen_pi: torch.Tensor,
        logp_rejected_pi: torch.Tensor,
        logp_chosen_ref: torch.Tensor,
        logp_rejected_ref: torch.Tensor,
    ) -> torch.Tensor:
        delta_pi = logp_chosen_pi - logp_rejected_pi
        delta_ref = logp_chosen_ref - logp_rejected_ref
        logits = self.beta * (delta_pi - delta_ref)
        return F.softplus(-logits).mean()  # -log(sigmoid(logits))

    def compute_loss(self, model, inputs: Dict[str, torch.Tensor], return_outputs: bool = False):
        # Ensure devices
        self._ensure_ref_device()

        device = self.model.device
        # Move required tensors to device if not already
        def to_dev(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        c_ids = to_dev(inputs["chosen_input_ids"])  # [B, T]
        c_attn = to_dev(inputs.get("chosen_attention_mask"))
        c_resp = to_dev(inputs["chosen_response_mask"])      # [B, T]

        r_ids = to_dev(inputs["rejected_input_ids"])  # [B, T]
        r_attn = to_dev(inputs.get("rejected_attention_mask"))
        r_resp = to_dev(inputs["rejected_response_mask"])    # [B, T]

        # Policy log-probs
        logp_c_pi = self._compute_response_logps(model, c_ids, c_attn, c_resp)
        logp_r_pi = self._compute_response_logps(model, r_ids, r_attn, r_resp)

        # Reference log-probs (no grad)
        with torch.no_grad():
            logp_c_ref = self._compute_response_logps(self.ref_model, c_ids, c_attn, c_resp)
            logp_r_ref = self._compute_response_logps(self.ref_model, r_ids, r_attn, r_resp)

        loss = self._dpo_loss(logp_c_pi, logp_r_pi, logp_c_ref, logp_r_ref)

        if return_outputs:
            outputs: Dict[str, Any] = {
                "loss": loss,
                "logp_chosen_pi": logp_c_pi.detach(),
                "logp_rejected_pi": logp_r_pi.detach(),
                "logp_chosen_ref": logp_c_ref,
                "logp_rejected_ref": logp_r_ref,
            }
            return loss, outputs
        return loss

    def train(self, *args, **kwargs):
        # Defer to base Trainer for the loop; our compute_loss implements DPO
        return super().train(*args, **kwargs)

class DPOPairwiseCollator:
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _build(self, prompts: List[str], responses: List[str]):
        texts = [p + r for p, r in zip(prompts, responses)]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prom_enc = self.tokenizer(prompts, add_special_tokens=False)
        prompt_lens = [len(ids) for ids in prom_enc["input_ids"]]

        B, T = enc["input_ids"].shape
        resp_mask = torch.zeros((B, T), dtype=torch.long)
        for i, pl in enumerate(prompt_lens):
            start = min(pl, T - 1)
            length = int(enc["attention_mask"][i].sum().item())
            resp_mask[i, start:length] = 1

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "response_mask": resp_mask,
        }

    def __call__(self, batch: List[dict]):
        prompts = [ex["prompt"] for ex in batch]
        chosens = [ex["chosen"] for ex in batch]
        rejecteds = [ex["rejected"] for ex in batch]

        c = self._build(prompts, chosens)
        r = self._build(prompts, rejecteds)

        return {
            "chosen_input_ids": c["input_ids"],
            "chosen_attention_mask": c["attention_mask"],
            "chosen_response_mask": c["response_mask"],
            "rejected_input_ids": r["input_ids"],
            "rejected_attention_mask": r["attention_mask"],
            "rejected_response_mask": r["response_mask"],
        }