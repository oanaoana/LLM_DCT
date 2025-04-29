import torch
import copy
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import llm_pswf
import pswf_utils as pswf
import dct_utils as dct  # Add this import

def warmup_device(device='cpu'):
    """
    Perform warmup operations to stabilize timing measurements on both CPU and GPU

    Args:
        device: 'cpu' or 'cuda' device to warm up
    """
    # Create tensors on the appropriate device
    dummy_a = torch.ones(1000, 1000, device=device)
    dummy_b = torch.ones(1000, 1000, device=device)

    # Run some compute-intensive operations
    for _ in range(5):
        _ = torch.matmul(dummy_a, dummy_b)
        _ = torch.relu(dummy_a)
        _ = torch.nn.functional.softmax(dummy_b, dim=1)

    # For CUDA devices, synchronize and clear cache
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

 # Define warm-up function
def warmup_model(model, tokenizer, device):
    """Warm up a specific model before benchmarking"""
    print(f"Warming up model on {device}...")
    model.eval()  # Ensure model is in eval mode

    # Warm up the device generally
    dummy = torch.ones(1000, 1000, device=device)
    for _ in range(5):
        _ = torch.matmul(dummy, dummy)

    # Warm up the specific model
    dummy_prompt = "This is a warm-up prompt."
    dummy_input = tokenizer(dummy_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        for _ in range(3):  # Run a few times to warm up
            _ = model.generate(**dummy_input, max_new_tokens=10)

    # For CUDA, synchronize and clear cache
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main():
    # 1. Load model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define custom sparsification parameters if needed
    def custom_sparsify_q(x):
        """Higher sparsity for queries - keep 98% of values"""
        percentile = 5
        threshold = torch.quantile(x.abs(), percentile / 100.0)
        mask = (x.abs() >= threshold).float()
        return x * mask

    def identity_fn(x):
        """Pass through function that does no sparsification"""
        return x

     # 3. Run comparison tests
    test_prompts = [
        "Once upon a time,",
        #"The quick brown fox",
        #"In a galaxy far far away,",
        #"The meaning of life is"
    ]

    print("=== Comparison of Original vs Sparsified Model ===\n")

    generated_texts = []
    for i, prompt in enumerate(test_prompts):
        print(f"=== Test Prompt {i+1}: '{prompt}' ===\n")

        # Load and prepare original model
        print("Setting up original model...")
        original_model = GPT2LMHeadModel.from_pretrained(model_name)
        original_model.to(device)
        original_model.eval()

        # Warm up original model
        warmup_model(original_model, tokenizer, device)

        # Test original model
        print("--- Original Model ---")
        original_results = llm_pswf.benchmark_inference(original_model, tokenizer, prompt, device=device.type)
        print(f"Generated: {original_results['text']}")
        print(f"Time: {original_results['time']:.4f} seconds")
        if original_results['memory'] is not None:
            print(f"Memory: {original_results['memory']:.2f} MB")
        print()

        # Clean up to save memory
        del original_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Load and prepare sparsified model
        print("Setting up sparsified model...")
        sparsified_model = GPT2LMHeadModel.from_pretrained(model_name)
        sparsified_model.to(device)

        # Get model dimensions
        hidden_size = sparsified_model.config.hidden_size  # 768 for GPT2-small
        num_heads = sparsified_model.config.n_head  # 12 for GPT2-small
        head_dim = hidden_size // num_heads  # 64 for GPT2-small

        # Calculate compression that maintains head compatibility
        compression_ratio = 0.5  # compress to 50%
        compressed_dim = int(hidden_size * compression_ratio)
        # Ensure compressed_dim is divisible by num_heads
        compressed_dim = (compressed_dim // num_heads) * num_heads
        #compressed_dim = hidden_size
        print(f"Using compressed dimension: {compressed_dim} (original: {hidden_size})")

        # Build the PSWF basis
        pswf_basis, pswf_inverse = pswf.build_pswf_basis(hidden_size, compressed_dim)

        # Define which layers to patch (without comma at the end of each line!)
        layers_to_patch = [0]  # Only first 6 layers

        # Define sparsification functions (without commas at the end!)
        #sparsify_q_fn = lambda x: pswf.pswf_project(x, pswf_basis)
        #sparsify_k_fn = lambda x: pswf.pswf_project(x, pswf_basis)
        #sparsify_v_fn = lambda x: pswf.pswf_project(x, pswf_basis)
        #backproject_fn = lambda x: pswf.pswf_backproject(x, pswf_inverse)

        # Build the DCT basis instead of PSWF
        dct_basis, dct_inverse = dct.build_dct_basis(hidden_size, compressed_dim)
        print(dct_basis)

        # Define sparsification functions
        sparsify_q_fn = lambda x: dct.dct_project(x, dct_basis)
        sparsify_k_fn = lambda x: dct.dct_project(x, dct_basis)
        sparsify_v_fn = lambda x: dct.dct_project(x, dct_basis)
        backproject_fn = lambda x: dct.dct_backproject(x, dct_inverse)

        # In main_pswf_perf.py, replace your sparsification functions with these for testing:
        #sparsify_q_fn = lambda x: x#[:, :compressed_dim]  # Simple truncation
        #sparsify_k_fn = lambda x: x#[:, :compressed_dim]  # Simple truncation
        #sparsify_v_fn = lambda x: x#[:, :compressed_dim]  # Simple truncation
        #backproject_fn = lambda x: x#torch.cat([x, torch.zeros(x.size(0), original_dim-compressed_dim, device=x.device)], dim=1)  # Pad with zeros

        # Apply attention sparsification
        sparsified_model = llm_pswf.apply_attention_sparsification(
            sparsified_model,
            sparsify_q_fn=sparsify_q_fn,  # Use named parameters for clarity
            sparsify_k_fn=sparsify_k_fn,
            sparsify_v_fn=sparsify_v_fn,
            compressed_dim=compressed_dim,
            layers_to_patch=layers_to_patch,
            backproject_fn=backproject_fn
        )
        sparsified_model.eval()

        # Warm up sparsified model
        warmup_model(sparsified_model, tokenizer, device)

        # Test sparsified model
        print("--- Sparsified Model (Identity Function) ---")
        sparse_results = llm_pswf.benchmark_inference(sparsified_model, tokenizer, prompt, device=device.type)
        print(f"Generated: {sparse_results['text']}")
        print(f"Time: {sparse_results['time']:.4f} seconds")
        if sparse_results['memory'] is not None:
            print(f"Memory: {sparse_results['memory']:.2f} MB")

        # Calculate metrics
        time_improvement = (original_results['time'] - sparse_results['time']) / original_results['time'] * 100
        print(f"Time improvement: {time_improvement:.2f}%")

        if original_results['memory'] is not None and sparse_results['memory'] is not None:
            memory_improvement = (original_results['memory'] - sparse_results['memory']) / original_results['memory'] * 100
            print(f"Memory improvement: {memory_improvement:.2f}%")

        # Store generated texts for BLEU comparison
        prompt_results = {
            "prompt": prompt,
            "original_text": original_results['text'],
            "sparse_text": sparse_results['text']
        }
        generated_texts.append(prompt_results)


        # Clean up
        del sparsified_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print("\n" + "="*50 + "\n")

    # 4. Run BLEU score comparison using stored texts
    try:
        import nltk
        nltk.download('punkt')
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize

        print("\n=== Output Quality Comparison ===\n")

        # Create smoothing function
        smoother = SmoothingFunction()

        for i, result in enumerate(generated_texts):
            prompt = result["prompt"]
            original_text = result["original_text"]
            sparse_text = result["sparse_text"]

            print(f"=== Test Prompt {i+1}: '{prompt}' ===\n")

            # Calculate BLEU score with smoothing
            reference = [word_tokenize(original_text)]
            candidate = word_tokenize(sparse_text)

            # BLEU-1 (unigrams only)
            bleu1 = sentence_bleu(reference, candidate,
                                weights=(1, 0, 0, 0),
                                smoothing_function=smoother.method1)

            # BLEU-2 (unigrams and bigrams)
            bleu2 = sentence_bleu(reference, candidate,
                                weights=(0.5, 0.5, 0, 0),
                                smoothing_function=smoother.method1)

            # Standard BLEU-4 with smoothing
            bleu4 = sentence_bleu(reference, candidate,
                                smoothing_function=smoother.method1)

            print(f"BLEU-1 Score: {bleu1:.4f}")
            print(f"BLEU-2 Score: {bleu2:.4f}")
            print(f"BLEU-4 Score: {bleu4:.4f}")

            # Print a sample of the texts
            print("\nOriginal text sample: ", original_text[:100], "...")
            print("Sparsified text sample: ", sparse_text[:100], "...")

            print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Error in BLEU calculation: {str(e)}")
        print("If using NLTK, install with: pip install nltk")

def debug_attention_implementation():
    """Debug the attention patching implementation"""
    print("\n=== DEBUGGING ATTENTION IMPLEMENTATION ===")

    # Load models
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Original model
    print("Setting up original model...")
    original_model = GPT2LMHeadModel.from_pretrained(model_name)
    original_model.to(device)
    original_model.eval()

    # Identity function
    identity_fn = lambda x: x

    # Create patched model with identity functions but no compression
    print("Creating patched model with identity functions...")
    patched_model = GPT2LMHeadModel.from_pretrained(model_name)
    patched_model.to(device)
    patched_model = llm_pswf.apply_attention_sparsification(
        patched_model,
        sparsify_q_fn=identity_fn,
        sparsify_k_fn=identity_fn,
        sparsify_v_fn=identity_fn,
        compressed_dim=None,  # No compression
        layers_to_patch=None,  # Patch all layers
        backproject_fn=None    # No backprojection needed
    )
    patched_model.eval()

    # Run basic forward pass comparison
    print("\nComparing forward pass outputs...")
    prompt = "Hello, world!"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Set consistent seed
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    with torch.no_grad():
        original_outputs = original_model(**inputs)
        original_logits = original_outputs.logits

    # Reset seed
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    with torch.no_grad():
        patched_outputs = patched_model(**inputs)
        patched_logits = patched_outputs.logits

    # Compare logits
    diff = (original_logits - patched_logits).abs()
    max_diff = diff.max().item()
    avg_diff = diff.mean().item()

    print(f"Max logits difference: {max_diff:.8f}")
    print(f"Mean logits difference: {avg_diff:.8f}")

    # Very detailed comparison - inspecting token probabilities
    print("\nToken probability differences:")
    for i in range(min(5, original_logits.size(1))):  # Check first few positions
        orig_probs = torch.softmax(original_logits[0, i], dim=-1)
        patch_probs = torch.softmax(patched_logits[0, i], dim=-1)

        # Get top tokens
        orig_top5 = torch.topk(orig_probs, 5)
        patch_top5 = torch.topk(patch_probs, 5)

        print(f"\nPosition {i} top tokens:")
        print("Original:", [f"{tokenizer.decode([idx.item()])}:{prob:.4f}" for idx, prob in zip(orig_top5.indices, orig_top5.values)])
        print("Patched: ", [f"{tokenizer.decode([idx.item()])}:{prob:.4f}" for idx, prob in zip(patch_top5.indices, patch_top5.values)])

    # Test generation
    print("\nTesting text generation...")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    original_text = generate_text(
        original_model, tokenizer, "Hello, world!", max_new_tokens=20, device=device.type
    )

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    patched_text = generate_text(
        patched_model, tokenizer, "Hello, world!", max_new_tokens=20, device=device.type
    )

    print(f"Original: {original_text}")
    print(f"Patched:  {patched_text}")
    print(f"Match: {'YES' if original_text == patched_text else 'NO'}")
    # Add this call to your debug function
    check_pswf_projection(768, 684)

def generate_text(model, tokenizer, prompt, max_new_tokens=50, do_sample=True, temperature=0.7, device="cuda"):
    """
    Generate text from a model based on a prompt

    Args:
        model: The model to use for generation
        tokenizer: The tokenizer for the model
        prompt: The text prompt to start generation
        max_new_tokens: Maximum number of new tokens to generate
        do_sample: Whether to use sampling (vs greedy decoding)
        temperature: Sampling temperature (higher = more random)
        device: Device to use for generation

    Returns:
        The generated text including the prompt
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Set pad token if not already set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def check_pswf_projection(hidden_size=768, compressed_dim=684):
    """Check if PSWF projection is working correctly"""
    print("\n=== Checking PSWF Projection ===")

    # Create random test data
    torch.manual_seed(42)  # For reproducibility
    test_data = torch.randn(4, 8, hidden_size)  # [batch, seq, hidden]
    print(f"Original data shape: {test_data.shape}")

    # Build PSWF basis
    pswf_basis, pswf_inverse = pswf.build_pswf_basis(hidden_size, compressed_dim)
    print(f"PSWF basis shape: {pswf_basis.shape}")
    print(f"PSWF inverse shape: {pswf_inverse.shape}")

    # Check that dimensions are compatible
    print(f"Basis inner dimension check: {pswf_basis.shape[1] == pswf_inverse.shape[0]}")
    print(f"Project: [{hidden_size}] -> [{compressed_dim}]")
    print(f"Backproject: [{compressed_dim}] -> [{hidden_size}]")

    # Flatten for projection
    test_flat = test_data.reshape(-1, hidden_size)
    print(f"Flattened data shape: {test_flat.shape}")

    # Project
    projected = pswf.pswf_project(test_flat, pswf_basis)
    print(f"Projected data shape: {projected.shape}")

    # Backproject
    backprojected = pswf.pswf_backproject(projected, pswf_inverse)
    print(f"Backprojected data shape: {backprojected.shape}")

    # Reshape back
    reshaped = projected.reshape(4, 8, -1)
    print(f"Reshaped data shape: {reshaped.shape}")

    # Calculate compression ratio
    actual_compression = reshaped.size(-1) / test_data.size(-1)
    print(f"Actual compression ratio: {actual_compression:.4f}")

    # Calculate reconstruction error
    reconstruction_error = (test_flat - backprojected).abs().mean().item()
    print(f"Reconstruction error: {reconstruction_error:.8f}")

    # Test if backprojects correctly
    if torch.allclose(test_flat, backprojected, atol=1e-5):
        print("✅ PSWF projection and backprojection working correctly!")
    else:
        print("❌ PSWF reconstruction has errors!")
        print(f"   Min error: {(test_flat - backprojected).abs().min().item():.8f}")
        print(f"   Max error: {(test_flat - backprojected).abs().max().item():.8f}")

# Add this call at the beginning of your main() function
debug_attention_implementation()

if __name__ == "__main__":
    main()