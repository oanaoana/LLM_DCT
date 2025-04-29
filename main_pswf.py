import torch
import copy
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import llm_pswf

def debug_attention_implementation(model_name="gpt2"):
    """
    Focus solely on debugging the attention implementation mismatch
    """
    print("=== DEBUGGING ATTENTION IMPLEMENTATION ===\n")

    # 1. Setup
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Create identity function for no sparsification
    def identity_fn(x):
        """Pass through function that does no sparsification"""
        return x

    # 3. Load original model
    print("Loading original model...")
    original_model = GPT2LMHeadModel.from_pretrained(model_name)
    original_model.to(device)
    original_model.eval()

    # 4. Create identical copy with patched attention
    print("Creating patched model with identity functions...")
    sparsified_model = copy.deepcopy(original_model)
    sparsified_model = llm_pswf.apply_attention_sparsification(
        sparsified_model,
        sparsify_q_fn=identity_fn,
        sparsify_k_fn=identity_fn,
        sparsify_v_fn=identity_fn
    )
    sparsified_model.eval()

    # 5. Run verification with detailed diagnostics
    print("\nRunning detailed verification...")
    debug_detailed_comparison(original_model, sparsified_model, tokenizer, device)

    # 6. Clean up
    del original_model
    del sparsified_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

def debug_detailed_comparison(original_model, patched_model, tokenizer, device):
    """
    Detailed layer-by-layer comparison of original vs patched model
    """
    # Set seeds for deterministic results
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # Create test input
    test_input = "This is a test input for detailed verification."
    inputs = tokenizer(test_input, return_tensors="pt").to(device)

    # 1. First check final outputs
    with torch.no_grad():
        original_outputs = original_model(**inputs)
        patched_outputs = patched_model(**inputs)

    final_diff = (original_outputs.logits - patched_outputs.logits).abs()
    print(f"\n=== FINAL OUTPUT DIFFERENCE ===")
    print(f"Max difference: {final_diff.max().item():.8f}")
    print(f"Mean difference: {final_diff.mean().item():.8f}")

    # 2. Register hooks to compare intermediate activations
    activation_original = {}
    activation_patched = {}

    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                activation_original[name] = output[0].detach()
            else:
                activation_original[name] = output.detach()
        return hook

    def get_activation_patched(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                activation_patched[name] = output[0].detach()
            else:
                activation_patched[name] = output.detach()
        return hook

    # Register hooks for transformer blocks
    for i, block in enumerate(original_model.transformer.h):
        block.register_forward_hook(get_activation(f'block_{i}'))

    for i, block in enumerate(patched_model.transformer.h):
        block.register_forward_hook(get_activation_patched(f'block_{i}'))

    # 3. Run forward pass again to collect activations
    with torch.no_grad():
        original_model(**inputs)
        patched_model(**inputs)

    # 4. Compare activations block by block
    print("\n=== BLOCK-BY-BLOCK COMPARISON ===")
    first_divergence_block = None
    for i in range(len(original_model.transformer.h)):
        block_name = f'block_{i}'
        if block_name in activation_original and block_name in activation_patched:
            diff = (activation_original[block_name] - activation_patched[block_name]).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            status = "✓" if max_diff < 1e-4 else "✗"
            print(f"Block {i}: {status} Max diff: {max_diff:.8f}, Mean diff: {mean_diff:.8f}")

            if max_diff >= 1e-4 and first_divergence_block is None:
                first_divergence_block = i

    # 5. Detailed diagnosis of problematic block
    if first_divergence_block is not None:
        print(f"\n=== DIAGNOSING BLOCK {first_divergence_block} ===")
        print("This is where the implementation first diverges significantly.")
        print("Check for issues in the following areas:")
        print("1. In your _attn method:")
        print("   - Ensure scaling factor uses sqrt(v.shape[-1])")
        print("   - Check attention mask and head mask handling")
        print("   - Verify softmax is applied along the right dimension")
        print("2. Ensure the patched attention is correctly handling all parameters")
    else:
        print("\nNo significant divergence found in intermediate layers.")
        print("The difference might be due to numerical precision issues.")

    # 6. Generate text to see if outputs are different
    print("\n=== TEXT GENERATION COMPARISON ===")
    test_prompt = "The quick brown fox"

    # Set the same seed for both generations
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    original_text = llm_pswf.generate_text(
        original_model, tokenizer, test_prompt, max_new_tokens=20, device=device.type
    )

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    patched_text = llm_pswf.generate_text(
        patched_model, tokenizer, test_prompt, max_new_tokens=20, device=device.type
    )

    print(f"Original: {original_text}")
    print(f"Patched:  {patched_text}")
    print(f"Match: {'YES' if original_text == patched_text else 'NO'}")

def main():
    # Just call the debug function
    debug_attention_implementation()

if __name__ == "__main__":
    main()