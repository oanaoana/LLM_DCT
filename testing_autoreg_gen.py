import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import llm_pswf

def test_autoregressive_generation():
    """Test token-by-token generation with patched attention"""
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Original model
    original_model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()

    # Identity function for no compression
    identity_fn = lambda x: x

    # Patched model with identity functions
    patched_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    patched_model = llm_pswf.apply_attention_sparsification(
        patched_model,
        sparsify_q_fn=identity_fn,
        sparsify_k_fn=identity_fn,
        sparsify_v_fn=identity_fn
    ).eval()

    # Run generation with several different inputs
    prompts = [
        "The quick brown fox",
        "Hello, my name is",
        "Once upon a time"
    ]

    for prompt in prompts:
        print(f"\nTesting prompt: '{prompt}'")

        # Set same seed for both
        torch.manual_seed(42)
        original_text = llm_pswf.generate_text(
            original_model, tokenizer, prompt, max_new_tokens=15, device=device.type
        )

        torch.manual_seed(42)
        patched_text = llm_pswf.generate_text(
            patched_model, tokenizer, prompt, max_new_tokens=15, device=device.type
        )

        print(f"Original: {original_text}")
        print(f"Patched:  {patched_text}")
        print(f"Match: {'✓' if original_text == patched_text else '✗'}")

        if original_text != patched_text:
            # Find where they first diverge
            min_len = min(len(original_text), len(patched_text))
            for i in range(min_len):
                if original_text[i] != patched_text[i]:
                    print(f"First divergence at position {i}: '{original_text[i]}' vs '{patched_text[i]}'")
                    break

if __name__ == "__main__":
    test_autoregressive_generation()