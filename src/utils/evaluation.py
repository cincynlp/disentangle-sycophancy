"""
Evaluation utilities for disentangle-sycophancy project.
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import List, Dict, Any, Optional, Tuple
import re


def compute_classification_metrics(
    y_true: List[int], 
    y_pred: List[int], 
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multi-class metrics
    
    Returns:
        Dictionary of computed metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    return metrics


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy from logits and labels.
    
    Args:
        logits: Model output logits
        labels: True labels
    
    Returns:
        Accuracy score
    """
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).float()
    return correct.mean().item()


def extract_answer_choice(text: str, choices: List[str] = None) -> Optional[str]:
    """
    Extract answer choice from generated text.
    
    Args:
        text: Generated text to extract answer from
        choices: List of valid choices (e.g., ['A', 'B', 'C', 'D'])
    
    Returns:
        Extracted choice or None if not found
    """
    if choices is None:
        choices = ['A', 'B', 'C', 'D']
    
    # Look for pattern like "The answer is A" or just "A"
    for choice in choices:
        patterns = [
            rf'\b{choice}\b',  # Standalone letter
            rf'answer is {choice}',  # "answer is A"
            rf'choice {choice}',  # "choice A"
            rf'\({choice}\)',  # (A)
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return choice
    
    return None


def evaluate_multiple_choice(
    predictions: List[str], 
    targets: List[str], 
    choices: List[str] = None
) -> Dict[str, float]:
    """
    Evaluate multiple choice predictions.
    
    Args:
        predictions: List of generated text predictions
        targets: List of correct answer choices
        choices: List of valid choices
    
    Returns:
        Dictionary with evaluation metrics
    """
    if choices is None:
        choices = ['A', 'B', 'C', 'D']
    
    extracted_preds = []
    valid_predictions = 0
    
    for pred in predictions:
        extracted = extract_answer_choice(pred, choices)
        extracted_preds.append(extracted)
        if extracted is not None:
            valid_predictions += 1
    
    # Compute accuracy only on valid predictions
    correct = 0
    total_valid = 0
    
    for pred, target in zip(extracted_preds, targets):
        if pred is not None:
            total_valid += 1
            if pred == target:
                correct += 1
    
    accuracy = correct / total_valid if total_valid > 0 else 0.0
    coverage = valid_predictions / len(predictions)
    
    return {
        'accuracy': accuracy,
        'coverage': coverage,
        'correct': correct,
        'total_valid': total_valid,
        'total': len(predictions)
    }


def compute_perplexity(loss: torch.Tensor) -> float:
    """
    Compute perplexity from loss.
    
    Args:
        loss: Cross-entropy loss tensor
    
    Returns:
        Perplexity score
    """
    return torch.exp(loss).item()


def batch_evaluate_generations(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    batch_size: int = 8
) -> Dict[str, float]:
    """
    Evaluate a batch of generations.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        responses: List of response strings
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary of evaluation metrics
    """
    total_loss = 0.0
    total_tokens = 0
    
    device = next(model.parameters()).device
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]
        
        # Tokenize batch
        full_texts = [p + r for p, r in zip(batch_prompts, batch_responses)]
        encodings = tokenizer(
            full_texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True
        )
        
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)
        
        # Create labels (only compute loss on response tokens)
        labels = input_ids.clone()
        for j, (prompt, response) in enumerate(zip(batch_prompts, batch_responses)):
            prompt_length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            labels[j, :prompt_length] = -100  # Ignore prompt tokens
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Count valid tokens (not ignored)
            valid_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens
    }


def compute_steering_metrics(
    baseline_outputs: List[str],
    steered_outputs: List[str],
    target_behavior: str = 'praise'
) -> Dict[str, float]:
    """
    Compute metrics for steering evaluation.
    
    Args:
        baseline_outputs: Outputs without steering
        steered_outputs: Outputs with steering
        target_behavior: Target behavior to measure
    
    Returns:
        Dictionary of steering metrics
    """
    # Simple heuristic-based evaluation
    # In practice, you might use a trained classifier
    
    def contains_praise(text: str) -> bool:
        praise_words = [
            'excellent', 'great', 'wonderful', 'amazing', 'fantastic',
            'outstanding', 'brilliant', 'perfect', 'superb', 'incredible'
        ]
        return any(word in text.lower() for word in praise_words)
    
    def contains_criticism(text: str) -> bool:
        criticism_words = [
            'terrible', 'awful', 'horrible', 'bad', 'worst',
            'disappointing', 'poor', 'inadequate', 'failure'
        ]
        return any(word in text.lower() for word in criticism_words)
    
    baseline_praise = sum(contains_praise(text) for text in baseline_outputs)
    steered_praise = sum(contains_praise(text) for text in steered_outputs)
    
    baseline_criticism = sum(contains_criticism(text) for text in baseline_outputs)
    steered_criticism = sum(contains_criticism(text) for text in steered_outputs)
    
    total = len(baseline_outputs)
    
    metrics = {
        'baseline_praise_rate': baseline_praise / total,
        'steered_praise_rate': steered_praise / total,
        'praise_increase': (steered_praise - baseline_praise) / total,
        'baseline_criticism_rate': baseline_criticism / total,
        'steered_criticism_rate': steered_criticism / total,
        'criticism_change': (steered_criticism - baseline_criticism) / total,
    }
    
    return metrics