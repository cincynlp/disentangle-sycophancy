"""
Dataset utilities for disentangle-sycophancy project.
"""
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Optional


class TextDataset(Dataset):
    """
    Dataset for handling prompt and response pairs.
    Pickle-safe Dataset placed at module scope to work with DataLoader(num_workers>0)
    on spawn-based platforms (e.g., macOS). Handles separate prompt and response strings,
    and constructs prompt/response masks accordingly.
    """

    def __init__(
        self,
        prompts: List[str],
        responses: List[str],
        tokenizer: AutoTokenizer,
        idxs: Optional[List[int]] = None,
    ):
        assert len(prompts) == len(
            responses
        ), "prompts and responses must have equal length"
        self.prompts = prompts
        self.responses = responses
        self.tokenizer = tokenizer
        if idxs is None:
            self.idxs = list(range(len(prompts)))
        else:
            self.idxs = idxs

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, i: int):
        idx = self.idxs[i]
        prompt = self.prompts[idx]
        response = self.responses[idx]
        
        # Create masks for different parts of the text
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, add_special_tokens=False
        ).input_ids[0]
        resp_ids = self.tokenizer(
            response, return_tensors="pt", truncation=True, add_special_tokens=False
        ).input_ids[0]
        enc = self.tokenizer(
            prompt + response,
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True,
        )
        input_ids = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)
        resp_start = input_ids.size(0) - resp_ids.size(0)
        prompt_end = resp_start
        
        # Create masks for different regions
        resp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        prompt_mask[:prompt_end] = True
        resp_mask[resp_start:] = True
        
        # Compute resp_eq_mask: portion of response at or after first '= '
        pos = response.find("= ")
        resp_eq_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Only try if substring found and tokenizer is fast
        if pos != -1 and getattr(self.tokenizer, "is_fast", False):
            # Tokenize response only, get offsets
            resp_enc = self.tokenizer(
                response,
                return_offsets_mapping=True,
                truncation=True,
                add_special_tokens=False,
            )
            offsets = resp_enc["offset_mapping"]
            rhs_char = pos + 2
            rhs_tok_start = None
            for i, (start, end) in enumerate(offsets):
                # Use token whose end offset > rhs_char (or start >= rhs_char)
                if end > rhs_char:
                    rhs_tok_start = i
                    break
            if rhs_tok_start is None:
                # If not found, fallback to resp_mask
                resp_eq_mask = resp_mask.clone()
            else:
                rhs_abs = resp_start + rhs_tok_start
                resp_eq_mask[rhs_abs:] = True
        else:
            resp_eq_mask = resp_mask.clone()
            
        # Statement-end (group-b) mask: last punctuation in the prompt ('.', '!', '?'), else last prompt token
        stmt_eos_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        # Compute prompt end (already have prompt_end)
        eos_char = -1
        for ch in (".", "!", "?"):
            j = prompt.rfind(ch)
            if j > eos_char:
                eos_char = j
        if eos_char == -1:
            stripped = prompt.rstrip()
            if stripped:
                eos_char = len(stripped) - 1
        if eos_char != -1 and getattr(self.tokenizer, "is_fast", False):
            p_enc = self.tokenizer(
                prompt,
                return_offsets_mapping=True,
                truncation=True,
                add_special_tokens=False,
            )
            offsets = p_enc["offset_mapping"]
            tok_idx = None
            for i_off, (start, end) in enumerate(offsets):
                if end > eos_char:
                    tok_idx = i_off
                    break
            if tok_idx is None and len(offsets) > 0:
                tok_idx = len(offsets) - 1
            if tok_idx is not None and tok_idx < prompt_end:
                stmt_eos_mask[tok_idx] = True
            elif prompt_end > 0:
                stmt_eos_mask[prompt_end - 1] = True
        else:
            if prompt_end > 0:
                stmt_eos_mask[prompt_end - 1] = True
                
        return (
            input_ids,
            attention_mask,
            resp_mask,
            resp_eq_mask,
            prompt_mask,
            stmt_eos_mask,
            idx,
        )