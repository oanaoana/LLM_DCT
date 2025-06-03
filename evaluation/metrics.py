import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(reference, candidate, n=4):
    """
    Calculate BLEU-n score between reference and candidate texts

    Args:
        reference (str): Reference text (ground truth)
        candidate (str): Candidate text to evaluate
        n (int): Maximum n-gram order to use (1-4)

    Returns:
        dict: Dictionary with BLEU-1 through BLEU-n scores
    """
    # Tokenize the texts into words
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()

    # Handle empty inputs
    if len(reference_tokens) == 0 or len(candidate_tokens) == 0:
        return {f'bleu{i}': 0.0 for i in range(1, n+1)}

    # Use NLTK's smoothing function for better handling of short segments
    smoothie = SmoothingFunction().method1

    # Calculate BLEU scores for different n-gram orders
    bleu_scores = {}

    for i in range(1, n+1):
        # Define weights for different n-gram orders
        # For BLEU-1, use weights [1.0, 0.0, 0.0, 0.0]
        # For BLEU-2, use weights [0.5, 0.5, 0.0, 0.0], etc.
        weights = [0.0] * 4
        for j in range(i):
            weights[j] = 1.0 / i

        # Calculate BLEU score with specified weights and smoothing
        try:
            score = sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                weights=weights,
                smoothing_function=smoothie
            )
        except Exception as e:
            print(f"Error calculating BLEU-{i}: {e}")
            score = 0.0

        bleu_scores[f'bleu{i}'] = score

    return bleu_scores

def evaluate_quality(original_model, compressed_model, tokenizer,
                   test_prompts, max_new_tokens=50, device=None):
    """
    Evaluate quality metrics between original and compressed models

    Args:
        original_model: The original uncompressed model
        compressed_model: The compressed model to evaluate
        tokenizer: Tokenizer for encoding/decoding text
        test_prompts: List of prompts to test generation quality
        max_new_tokens: Maximum number of tokens to generate
        device: Device to run models on ('cuda', 'cpu')

    Returns:
        dict: Dictionary with quality metrics and generated texts
    """
    # Determine device automatically if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Ensure the models are in evaluation mode
    original_model.eval()
    compressed_model.eval()

    # Models should already be on the right device from main_final.py
    # But just to be safe, check the device and move if needed
    if next(original_model.parameters()).device != device:
        original_model = original_model.to(device)
    if next(compressed_model.parameters()).device != device:
        compressed_model = compressed_model.to(device)

    # Initialize result dictionaries
    results = {
        'bleu1': [],
        'bleu2': [],
        'bleu3': [],
        'bleu4': [],
        'texts': []
    }

    # Process each prompt
    for i, prompt in enumerate(test_prompts):
        print(f"\nEvaluating prompt {i+1}/{len(test_prompts)}: '{prompt[:30]}...'")

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate text with both models using the same seed for fair comparison
        with torch.no_grad():
            # Set seed for reproducibility
            torch.manual_seed(42)
            original_outputs = original_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

            # Reset seed to same value
            torch.manual_seed(42)
            compressed_outputs = compressed_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the generated texts
        original_text = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
        compressed_text = tokenizer.decode(compressed_outputs[0], skip_special_tokens=True)

        # Store the full texts
        results['texts'].append({
            'prompt': prompt,
            'original': original_text,
            'compressed': compressed_text
        })

        # Calculate BLEU scores (comparing compressed output to original output)
        # We only evaluate the newly generated portion, not including the prompt
        prompt_len = len(prompt)
        original_generated = original_text[prompt_len:].strip()
        compressed_generated = compressed_text[prompt_len:].strip()

        # If either generated text is empty, skip BLEU calculation
        if not original_generated or not compressed_generated:
            print("  Warning: Empty generation detected, skipping BLEU calculation")
            continue

        bleu_scores = calculate_bleu(original_generated, compressed_generated)

        # Store BLEU scores
        for metric, score in bleu_scores.items():
            results[metric].append(score)

        # Print progress
        print(f"  BLEU-1: {bleu_scores['bleu1']:.4f}, BLEU-4: {bleu_scores['bleu4']:.4f}")
        print(f"  Original: '{original_generated[:50]}...'")
        print(f"  Compressed: '{compressed_generated[:50]}...'")

    # Calculate average metrics at the end
    for metric in ['bleu1', 'bleu2', 'bleu3', 'bleu4']:
        if results[metric]:  # Check if we have any scores
            results[f'avg_{metric}'] = sum(results[metric]) / len(results[metric])
        else:
            results[f'avg_{metric}'] = 0.0

    return results


def compare_models_with_human_eval(original_model, compressed_model, tokenizer, test_prompts, device='cuda'):
    """
    Generate outputs from both models to assist human evaluation

    Args:
        original_model: The original uncompressed model
        compressed_model: The compressed model to evaluate
        tokenizer: Tokenizer for encoding/decoding text
        test_prompts: List of prompts to test generation quality
        device: Device to run models on ('cuda', 'cpu')

    Returns:
        list: List of dictionaries with prompt and model outputs for human evaluation
    """
    results = []

    # Process each prompt
    for i, prompt in enumerate(test_prompts):
        print(f"\nGenerating for prompt {i+1}/{len(test_prompts)}")

        # Generate outputs
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = {}
        with torch.no_grad():
            # Original model
            torch.manual_seed(42)
            original_output = original_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            outputs['original'] = tokenizer.decode(original_output[0], skip_special_tokens=True)

            # Compressed model
            torch.manual_seed(42)
            compressed_output = compressed_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            outputs['compressed'] = tokenizer.decode(compressed_output[0], skip_special_tokens=True)

        # Add to results
        results.append({
            'prompt': prompt,
            'original': outputs['original'],
            'compressed': outputs['compressed']
        })

        print(f"Prompt: {prompt}")
        print(f"Original: {outputs['original']}")
        print(f"Compressed: {outputs['compressed']}")

    return results


def calculate_perplexity(model, tokenizer, text, device='cuda'):
    """
    Calculate perplexity of a model on given text

    Perplexity = exp(average negative log-likelihood per token)

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for encoding text
        text: Text to calculate perplexity on
        device: Device to run model on

    Returns:
        float: Perplexity score (lower is better)
    """
    # Encode the text
    encodings = tokenizer(text, return_tensors='pt').to(device)

    # Get sequence length for averaging later
    seq_len = encodings.input_ids.size(1)

    # Create targets by shifting inputs (next word prediction)
    targets = encodings.input_ids.clone()

    # Calculate loss with model
    with torch.no_grad():
        outputs = model(**encodings, labels=targets)
        neg_log_likelihood = outputs.loss.item()

    # Perplexity is the exponentiated average negative log-likelihood per token
    perplexity = torch.exp(torch.tensor(neg_log_likelihood)).item()

    return perplexity


def evaluate_perplexity(model, tokenizer, test_texts, device='cuda'):
    """
    Evaluate model perplexity on a set of test texts

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for encoding text
        test_texts: List of texts to evaluate perplexity on
        device: Device to run model on

    Returns:
        dict: Dictionary with perplexity scores
    """
    # Ensure the model is in evaluation mode
    model.eval()
    model = model.to(device)

    # Initialize results
    results = {
        'perplexities': [],
        'texts': test_texts
    }

    # Calculate perplexity for each text
    for i, text in enumerate(test_texts):
        try:
            perplexity = calculate_perplexity(model, tokenizer, text, device)
            results['perplexities'].append(perplexity)
            print(f"Text {i+1}/{len(test_texts)}: Perplexity = {perplexity:.4f}")
        except Exception as e:
            print(f"Error calculating perplexity for text {i+1}: {e}")
            results['perplexities'].append(float('inf'))

    # Calculate average perplexity
    valid_perplexities = [p for p in results['perplexities'] if p != float('inf')]
    if valid_perplexities:
        results['avg_perplexity'] = sum(valid_perplexities) / len(valid_perplexities)
    else:
        results['avg_perplexity'] = float('inf')

    print(f"\nAverage perplexity: {results['avg_perplexity']:.4f}")

    return results