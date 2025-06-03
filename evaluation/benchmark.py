import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils.profiling import Timer, MemoryTracker, FlopCounter, profile_execution, BenchmarkSuite
from models.sparsification import apply_attention_sparsification
import transforms.dct as dct

def run_benchmarks():
    # Load model and tokenizer
    model_name = "gpt2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load original model
    print("Loading original model...")
    original_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    original_model.eval()

    # Create compressed model
    print("Creating compressed model...")
    compressed_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    # Get model dimensions
    hidden_size = compressed_model.config.hidden_size
    num_heads = compressed_model.config.n_head

    # Set compression ratio
    compression_ratio = 0.5
    compressed_dim = int(hidden_size * compression_ratio)
    compressed_dim = (compressed_dim // num_heads) * num_heads

    # Build DCT basis
    dct_basis, dct_inverse = dct.build_dct_basis(hidden_size, compressed_dim)

    # Set up compression functions
    sparsify_fn = lambda x: dct.dct_project(x, dct_basis)
    backproject_fn = lambda x: dct.dct_backproject(x, dct_inverse)

    # Apply compression to the model
    compressed_model = apply_attention_sparsification(
        compressed_model,
        sparsify_q_fn=sparsify_fn,
        sparsify_k_fn=sparsify_fn,
        sparsify_v_fn=sparsify_fn,
        backproject_fn=backproject_fn,
        compressed_dim=compressed_dim
    )
    compressed_model.eval()

    # Sample prompts for benchmarking
    prompts = [
        "The best way to predict the future is to",
        "Once upon a time in a land far, far away",
        "The key to successful machine learning is"
    ]

    # Create benchmark suites
    print("\nBenchmarking original model...")
    original_benchmark = BenchmarkSuite(original_model, tokenizer, device=device)
    original_results = original_benchmark.run_full_benchmark(prompts)

    print("\nBenchmarking compressed model...")
    compressed_benchmark = BenchmarkSuite(compressed_model, tokenizer, device=device)
    compressed_results = compressed_benchmark.run_full_benchmark(prompts)

    # Compare results
    print("\n=== Performance Comparison ===")

    # Latency comparison
    original_latency = original_results["latency"][0]["stats"]["mean_time"]
    compressed_latency = compressed_results["latency"][0]["stats"]["mean_time"]
    latency_speedup = original_latency / compressed_latency

    print(f"Latency speedup: {latency_speedup:.2f}x")

    # Memory comparison
    original_memory = original_results["memory"]["memory_gpu_mb"]
    compressed_memory = compressed_results["memory"]["memory_gpu_mb"]
    memory_reduction = original_memory / compressed_memory

    print(f"Memory reduction: {memory_reduction:.2f}x")

    # Throughput comparison
    original_throughput = original_results["throughput"][0]["tokens_per_second"]
    compressed_throughput = compressed_results["throughput"][0]["tokens_per_second"]
    throughput_speedup = compressed_throughput / original_throughput

    print(f"Throughput improvement: {throughput_speedup:.2f}x")

    return {
        "original": original_results,
        "compressed": compressed_results,
        "comparison": {
            "latency_speedup": latency_speedup,
            "memory_reduction": memory_reduction,
            "throughput_speedup": throughput_speedup
        }
    }

# Individual profiling example
def profile_individual_operations():
    model_name = "gpt2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    prompt = "The universe is a very big place, perhaps the biggest."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Using the Timer
    with Timer("Generate 20 tokens"):
        outputs = model.generate(**inputs, max_new_tokens=20)

    # Using MemoryTracker
    with MemoryTracker(device):
        outputs = model.generate(**inputs, max_new_tokens=50)

    # Profile single forward pass
    with profile_execution("Model forward pass", device, {"model": model, "seq_len": inputs.input_ids.shape[1]}):
        outputs = model(**inputs)

    # Estimate FLOPs for the model
    flops = FlopCounter.model_flops(model, seq_len=inputs.input_ids.shape[1])
    print(f"Estimated model GFLOPs: {flops['total_gflops']:.2f}")
    print(f"Attention GFLOPs: {flops['attention']['total_gflops']:.2f}")

if __name__ == "__main__":
    # Run individual profiling
    profile_individual_operations()

    # Run full benchmark comparison
    results = run_benchmarks()

def benchmark_inference(model, tokenizer, prompt, max_new_tokens=50, device=None, num_runs=3, warmup=2):
    """
    Benchmark inference time and memory usage with proper warmup

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for encoding text
        prompt: Text prompt for generation
        max_new_tokens: Maximum number of tokens to generate
        device: Device to run on (cuda or cpu)
        num_runs: Number of timed runs to average
        warmup: Number of warmup runs before timing

    Returns:
        dict: Benchmark results
    """
    # Determine device automatically if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Make sure model is on the correct device and in eval mode
    model = model.to(device)
    model.eval()

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    # Add generation parameters
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id
    }

    # Warmup runs (important for GPU performance measurements)
    print(f"  Running {warmup} warmup iterations...", end='', flush=True)
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, **gen_kwargs)
    print(" done")

    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Multiple timed runs
    times = []
    memory_usage = []
    generated_text = None

    print(f"  Running {num_runs} timed iterations...", end='', flush=True)
    for run in range(num_runs):
        # Clear cache between runs if using CUDA
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated()
        else:
            start_mem = None

        # Time the generation
        import time
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Synchronize before stopping timer
        if device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # Track memory if using CUDA
        if device.type == 'cuda':
            end_mem = torch.cuda.memory_allocated()
            memory_used = (end_mem - start_mem) / 1024**2  # Convert to MB
            memory_usage.append(memory_used)

        # Record time and save output from last run
        times.append(end_time - start_time)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(" done")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    # For memory, take the average if available
    if device.type == 'cuda' and memory_usage:
        avg_memory = sum(memory_usage) / len(memory_usage)
    else:
        avg_memory = None

    # Calculate tokens per second
    tokens_generated = len(tokenizer.encode(generated_text)) - len(tokenizer.encode(prompt))
    tokens_per_second = tokens_generated / avg_time

    results = {
        "text": generated_text,
        "time": avg_time,
        "time_min": min_time,
        "time_max": max_time,
        "time_std": std_time,
        "memory": avg_memory,
        "tokens_per_second": tokens_per_second,
        "tokens_generated": tokens_generated,
        "device": str(device)
    }

    return results
