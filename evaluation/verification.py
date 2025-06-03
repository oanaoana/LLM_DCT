import torch

def verify_attention_implementation(original_model, sparsified_model, tokenizer, device):
    """
    Verify that patched attention implementation produces the same results as original
    when using identity functions (no sparsification)
    """
    print("\n=== VERIFYING ATTENTION IMPLEMENTATION ===")

    # Set seeds for deterministic results
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # Create identical input
    test_input = "This is a test input for verification."
    inputs = tokenizer(test_input, return_tensors="pt").to(device)

    # Get outputs from both models
    with torch.no_grad():
        # Original model
        original_outputs = original_model(**inputs)
        original_logits = original_outputs.logits

        # Patched model
        sparsified_outputs = sparsified_model(**inputs)
        sparsified_logits = sparsified_outputs.logits

    # Compare logits (output predictions)
    diff = (original_logits - sparsified_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Maximum absolute difference: {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")

    # Check if they're close enough
    is_equal = max_diff < 1e-4  # Using a slightly larger threshold for numerical stability
    print(f"Implementations match: {'YES' if is_equal else 'NO'}")

    # For non-matching implementations, show where differences occur
    if not is_equal:
        print("\nDifferences in attention implementation detected!")
        print("This means your patched attention may not be mathematically equivalent to the original.")
        print("Check your patched implementation for:")
        print("1. Missing or incorrect _attn method call")
        print("2. Different scaling in attention computation")
        print("3. Handling of attention mask and head mask")

    return is_equal

def check_transform_accuracy(input_dim, compressed_dim, transform_type='dct'):
    """Check reconstruction accuracy for various transforms"""
    # Implementation...