"""
Generation utilities for disentangle-sycophancy project.
"""
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Optional, Tuple, Any
import accelerate


def _generate_next(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    max_new_tokens: int = 1,
    pad_token_id: Optional[int] = None,
    end_on: Optional[str] = None,
    add_prefix_space: bool = True,
    **kwargs
) -> torch.Tensor:
    """
    Generate text continuation with early stopping.
    
    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer for the model
        input_ids: Input token IDs tensor
        temperature: Sampling temperature
        max_new_tokens: Maximum number of new tokens to generate
        pad_token_id: Token ID to use for padding
        end_on: String to stop generation on
        add_prefix_space: Whether to add a prefix space when tokenizing end_on
        **kwargs: Additional generation arguments
    
    Returns:
        Generated token IDs tensor
    """
    with torch.no_grad():
        orig_len = input_ids.size(1)
        
        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        
        # Prepare stop tokens if end_on is specified
        stop_token_ids = None
        if end_on is not None:
            if add_prefix_space and not end_on.startswith(" "):
                end_on = " " + end_on
            stop_tokens = tokenizer(end_on, add_special_tokens=False).input_ids
            if stop_tokens:
                stop_token_ids = torch.tensor(stop_tokens, device=input_ids.device)
        
        # Generate tokens one by one with early stopping
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for early stopping
            if stop_token_ids is not None:
                # Check if we've generated the stop sequence
                seq_len = generated_ids.size(1)
                stop_len = stop_token_ids.size(0)
                
                if seq_len >= orig_len + stop_len:
                    recent_tokens = generated_ids[:, -stop_len:]
                    if torch.equal(recent_tokens[0], stop_token_ids):
                        break
        
        return generated_ids


def generate_on_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch: Tuple[torch.Tensor, ...],
    max_new_tokens: int = 1,
    temperature: float = 1.0,
    **kwargs
) -> torch.Tensor:
    """
    Generate text for a batch of inputs.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        batch: Batch tuple containing input_ids and attention_mask
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        **kwargs: Additional generation arguments
    
    Returns:
        Generated token IDs tensor
    """
    input_ids, attention_mask = batch[0], batch[1]
    
    # Move to model device if needed
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **kwargs
        )
    
    return outputs


def predict_on_batch(
    model: PreTrainedModel,
    batch: Tuple[torch.Tensor, ...],
    **kwargs
) -> torch.Tensor:
    """
    Get model predictions on a batch.
    
    Args:
        model: The model to use for prediction
        batch: Batch tuple with input data
        **kwargs: Additional model arguments
    
    Returns:
        Model outputs tensor
    """
    input_ids, attention_mask = batch[0], batch[1]
    
    # Move to model device if needed
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    return outputs


def setup_generation_config(
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
) -> dict:
    """
    Set up generation configuration dictionary.
    
    Args:
        tokenizer: The tokenizer
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling parameter
        do_sample: Whether to use sampling
    
    Returns:
        Generation configuration dictionary
    """
    config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    return config