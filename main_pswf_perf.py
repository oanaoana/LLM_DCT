import torch
import copy
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import llm_pswf
import pswf_utils as pswf

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
        "The quick brown fox",
        "In a galaxy far far away,",
        "The meaning of life is"
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
        compression_ratio = 1.0  # compress to 50%
        compressed_dim = int(hidden_size * compression_ratio)
        # Ensure compressed_dim is divisible by num_heads
        compressed_dim = (compressed_dim // num_heads) * num_heads
        print(f"Using compressed dimension: {compressed_dim} (original: {hidden_size})")

        # Build the PSWF basis
        pswf_basis, pswf_inverse = pswf.build_pswf_basis(hidden_size, compressed_dim)

        # Define which layers to patch (without comma at the end of each line!)
        layers_to_patch = None #[0]  # Only first 6 layers

        # Define sparsification functions (without commas at the end!)
        sparsify_q_fn = lambda x: pswf.pswf_project(x, pswf_basis)
        sparsify_k_fn = lambda x: pswf.pswf_project(x, pswf_basis)
        sparsify_v_fn = lambda x: pswf.pswf_project(x, pswf_basis)
        backproject_fn = lambda x: pswf.pswf_project(x, pswf_inverse)

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

if __name__ == "__main__":
    main()