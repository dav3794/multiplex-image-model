#!/usr/bin/env python3
# get_model.py
# -*- coding: utf-8 -*-


import os
import time
import torch
import numpy as np

from modules_vit import MultiplexAutoencoder

from ruamel.yaml import YAML
from fvcore.nn import FlopCountAnalysis
from deepspeed.accelerator import get_accelerator
from deepspeed.profiling.flops_profiler import get_model_profile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_yaml(config_path):
    yaml = YAML(typ='safe')
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    num_channels_model = 265 
    input_size = tuple(config.get('input_image_size', [128, 128]))
    model = MultiplexAutoencoder(
        num_channels=num_channels_model,
        encoder_config=config['encoder'],
        decoder_config=config['decoder'],
        input_image_size=input_size
    )
    
    return model, input_size

def profile_model(config_path):
    # Loading
    try:
        model, input_size = load_model_from_yaml(config_path)
    except Exception as e:
        print(f"[ERROR:] loading {config_path}: {e}")
        return None

    model.to(device)
    model.eval()

    H, W = input_size
    bench_channels = 40
    batch_size = 1
    dummy_input = torch.randn(batch_size, bench_channels, H, W).to(device)
    dummy_indices = torch.arange(bench_channels).unsqueeze(0).repeat(batch_size, 1).to(device)
    inputs = (dummy_input, dummy_indices, dummy_indices)
    
    # Params calculation
    total_params = 0
    hk_params = 0
    
    for name, p in model.named_parameters():
        n = p.numel()
        total_params += n

        if "hyperkernel" in name.lower():
            hk_params += n

    if hk_params == 0:
        print(f"[WARNING:] No parameters with 'hyperkernel' in name found for {config_path}!")

    ratio = 40.0 / 265.0
    active_params = (total_params - hk_params) + (hk_params * ratio)
    
    params_active_m = active_params / 1e6
    params_total_m = total_params / 1e6

    # GFLOPS
    gflops = 0.0
    try:
        flops_fv = FlopCountAnalysis(model, inputs)
        flops_fv.unsupported_ops_warnings(False) 
        gflops = flops_fv.total() / 1e9
    except Exception:
        try:
                with get_accelerator().device(0):
                    flops_ds, _, _ = get_model_profile(model, inputs, print_profile=False, detailed=False)
                gflops = flops_ds / 1e9
        except:
            pass

    # Peak memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(*inputs)

    peak_mem_gb = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0

    # Time
    outer_loops = 100
    inner_loops = 100
    block_timings = []
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(*inputs)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(outer_loops):
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                for _ in range(inner_loops):
                    _ = model(*inputs)

                end_event.record()
                torch.cuda.synchronize()
                block_timings.append(start_event.elapsed_time(end_event))
            else:
                start_t = time.time()
                for _ in range(inner_loops):
                    _ = model(*inputs)

                block_timings.append((time.time() - start_t) * 1000)

    timings = np.array(block_timings)
    avg_ms = np.mean(timings)
    std_ms = np.std(timings)
    arch_name = os.path.basename(config_path).replace("_config.yaml", "").replace(".yaml", "")

    del model
    torch.cuda.empty_cache()

    return {
        "Architecture": arch_name,
        "Active Params (M)": params_active_m,
        "Total Params (M)": params_total_m,
        "GFLOPs": gflops,
        "Time (ms)": avg_ms,
        "Std (ms)": std_ms,
        "Mem (GB)": peak_mem_gb
    }

if __name__ == "__main__":
    configs_to_measure = [
        "swin_small_config.yaml",
        "vitS_config.yaml",
        "vitL_config.yaml",
        "vitM_config.yaml",
        "Swin_benchmark.yaml"
    ]
    
    results = []
    print(f"Profiling {len(configs_to_measure)} models on {device}.")
    print("Methodology: Heuristic Active Params (Total - HK + HK*40/265). Time sum for 100 inputs.\n")

    for config_file in configs_to_measure:
        if os.path.exists(config_file):
            print(f"-> Benchmarking: {config_file} ...")
            res = profile_model(config_file)
            if res:
                results.append(res)
        else:
            print(f"Skipping (not found): {config_file}")

    # SUMMARY
    print("\n" + "="*110)
    header = f"{'Architecture':<25} | {'Active (M)':<12} | {'Total (M)':<12} | {'GFLOPs':<10} | {'Time (ms)*':<18} | {'Mem (GB)':<10}"
    print(header)
    print("-" * 110)
    
    for r in results:
        time_str = f"{r['Time (ms)']:.2f} ± {r['Std (ms)']:.2f}"
        
        row = f"{r['Architecture']:<25} | {r['Active Params (M)']:<12.2f} | {r['Total Params (M)']:<12.2f} | {r['GFLOPs']:<10.2f} | {time_str:<18} | {r['Mem (GB)']:<10.3f}"
        print(row)
    print("=" * 110)
    print("* Time (ms) represents the total duration to process 100 inputs sequentially.")
    print("* Active (M) = (Total - HK) + HK * (40/265)")
    print("\n")
