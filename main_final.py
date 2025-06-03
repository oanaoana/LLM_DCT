import torch
import argparse
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from models.sparsification import apply_attention_sparsification
import transforms.dct as dct
import transforms.pswf as pswf
from evaluation.benchmark import benchmark_inference
from evaluation.metrics import evaluate_quality
from evaluation.verification import verify_attention_implementation

def main():
    parser = argparse.ArgumentParser(description="Compress LLM attention with transforms")
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--transform', type=str, default='dct', choices=['dct', 'pswf'],
                        help='Transform type')
    parser.add_argument('--compression', type=float, default=0.8,
                        help='Compression ratio (0.0-1.0)')
    parser.add_argument('--layers', type=str, default='all',
                        help='Layers to patch (comma-separated or "all")')
    parser.add_argument('--verify', action='store_true', help='Run verification')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--evaluate', action='store_true', help='Run quality evaluation')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode even if GPU is available')

    args = parser.parse_args()

    # Set device with respect to --cpu flag
    if args.cpu:
        device = torch.device("cpu")
        print("Forcing CPU mode as requested")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            print("Using GPU for inference")
        else:
            print("Using CPU for inference (no GPU available)")

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load original model
    original_model = GPT2LMHeadModel.from_pretrained(args.model).to(device)

    # Parse layers to patch
    if args.layers == 'all':
        layers_to_patch = None  # All layers
    else:
        layers_to_patch = [int(x) for x in args.layers.split(',')]

    # Get model dimensions
    hidden_size = original_model.config.hidden_size
    num_heads = original_model.config.n_head

    # Calculate compressed dimension
    compressed_dim = int(hidden_size * args.compression)
    compressed_dim = (compressed_dim // num_heads) * num_heads

    # Build transform basis
    if args.transform == 'dct':
        basis, inverse = dct.build_dct_basis(hidden_size, compressed_dim)
        project_fn = lambda x: dct.dct_project(x, basis)
        backproject_fn = lambda x: dct.dct_backproject(x, inverse)
    else:  # pswf
        basis, inverse = pswf.build_pswf_basis(hidden_size, compressed_dim)
        project_fn = lambda x: pswf.pswf_project(x, basis)
        backproject_fn = lambda x: pswf.pswf_backproject(x, inverse)

    # Create compressed model
    compressed_model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    compressed_model = apply_attention_sparsification(
        compressed_model,
        sparsify_q_fn=project_fn,
        sparsify_k_fn=project_fn,
        sparsify_v_fn=project_fn,
        backproject_fn=backproject_fn,
        compressed_dim=compressed_dim,
        layers_to_patch=layers_to_patch
    )

    # Run verification if requested
    if args.verify:
        # Create identity model for verification
        identity_model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
        identity_model = apply_attention_sparsification(
            identity_model,
            sparsify_q_fn=lambda x: x,
            sparsify_k_fn=lambda x: x,
            sparsify_v_fn=lambda x: x,
            layers_to_patch=layers_to_patch
        )
        verify_attention_implementation(original_model, identity_model, tokenizer, device)

    # Run benchmarks if requested
    if args.benchmark:
        prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "In a galaxy far, far away, the forces of good and evil battled for supremacy.",
            "Once upon a time in a land far away, there lived a princess who"
        ]

        print("\n=== Performance Benchmark ===")
        for prompt in prompts:
            orig_results = benchmark_inference(original_model, tokenizer, prompt, device=device)
            comp_results = benchmark_inference(compressed_model, tokenizer, prompt, device=device)

            print(f"\nPrompt: {prompt[:30]}...")
            print(f"Original model time: {orig_results['time']:.4f}s")
            print(f"Compressed model time: {comp_results['time']:.4f}s")
            print(f"Speedup: {orig_results['time'] / comp_results['time']:.2f}x")
            print(f"Running on: {orig_results['device']}")

            if orig_results['memory'] is not None:
                print(f"Memory reduction: {comp_results['memory'] / orig_results['memory']:.2f}x")
            else:
                print("Memory usage data not available on CPU")

    # Run quality evaluation if requested
    if args.evaluate:
        test_prompts = [
            "The meaning of life is",
            "Once upon a time there was",
            "The best way to learn programming is",
            "In five years, artificial intelligence will",
            "The solution to climate change involves"
        ]

        # Pass the device parameter to evaluate_quality
        results = evaluate_quality(original_model, compressed_model, tokenizer, test_prompts, device=device)

        # Calculate average BLEU scores if they're lists
        avg_bleu1 = sum(results['bleu1']) / len(results['bleu1']) if isinstance(results['bleu1'], list) else results['bleu1']
        avg_bleu4 = sum(results['bleu4']) / len(results['bleu4']) if isinstance(results['bleu4'], list) else results['bleu4']

        print("\n=== Quality Evaluation ===")
        print(f"Average BLEU-1 Score: {avg_bleu1:.4f}")
        print(f"Average BLEU-4 Score: {avg_bleu4:.4f}")

        # For individual BLEU scores
        print("\nBLEU Scores by Prompt:")
        for i, (b1, b4) in enumerate(zip(results['bleu1'], results['bleu4'])):
            print(f"Prompt {i+1}: BLEU-1: {b1:.4f}, BLEU-4: {b4:.4f}")

        # Print some examples
        print("\n=== Generation Examples ===")
        for i, prompt in enumerate(test_prompts[:2]):  # Show first 2 examples
            print(f"\nPrompt: {prompt}")
            print(f"Original: {results['texts'][i]['original']}")
            print(f"Compressed: {results['texts'][i]['compressed']}")

if __name__ == "__main__":
    # Force CPU mode to avoid CUDA errors
    #sys.argv.append('--cpu')  # Force CPU mode
    sys.argv.append('--evaluate')
    #sys.argv.append('--transform')
    #sys.argv.append('--benchmark')
    sys.argv.append('--verify')
    main()