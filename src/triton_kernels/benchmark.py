import triton
import torch
from torch import nn, Tensor
from importlib import import_module

from triton_kernels import nn as triton_nn
from triton_kernels.nn import KERNELS
from triton_kernels.utils import get_device
from copy import deepcopy

DEVICE = get_device()



default_base_benchmark_kwargs = {
        "x_names":['N'],  # argument names to use as an x-axis for the plot
        "x_vals":[128 * i for i in range(2, 100, 10)],  # different possible values for `x_name`
        "line_arg":'provider',  # argument name whose value corresponds to a different line in the plot
        "line_vals":['triton', 'torch'],  # possible values for `line_arg``
        "line_names":["Triton", "Torch"],  # label name for the lines
        # "styles":[('blue', '-'), ('green', '-')],  # line styles
        "ylabel":"GB/s",  # label name for the y-axis
        "plot_name":"softmax",  # name for the plot. Used also as a file name for saving the plot.
        "args":{'M': 4096} # values for function arguments not in `x_names` and `y_name`
}






def measure_memory(f, *args, **kwargs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    f(*args, **kwargs)  # run your function once

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e6  # MB
    return peak

def benchmark_kernel(fwd, base_benchmark_kwargs = default_base_benchmark_kwargs):
    
    def bwd(x, provider):
        x.requires_grad = True
        out = fwd(x, provider)
        loss = out.sum()
        loss.backward()


    MAP_FWD_BKW = {
        "fwd" : fwd,
        "bwd" : bwd,
    }
    
    configs = []
    for bench_kind in ['timing', 'memory']:
        for mode in ['fwd', 'bwd']:
            _kwargs = deepcopy(base_benchmark_kwargs)
            _kwargs['args'].update({"mode" : mode})
            _kwargs['args'].update({"bench_kind" : bench_kind})
            _kwargs['ylabel'] = 'ms' if bench_kind == 'timing' else 'MB'
            _kwargs['plot_name'] += f' - {bench_kind} - {mode}'
            configs.append(triton.testing.Benchmark(**_kwargs))

    @triton.testing.perf_report(configs)
    def benchmark(M, N, provider, mode, bench_kind):
        x = torch.randn(M, N, device=DEVICE, dtype=torch.bfloat16)
        stream = getattr(torch, DEVICE.type).Stream()
        getattr(torch, DEVICE.type).set_stream(stream)
        if bench_kind == "timing":
            ms = triton.testing.do_bench(lambda: MAP_FWD_BKW[mode](x, provider))
            # gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
            return ms #gbps(ms)
        elif bench_kind == "memory":
            mem_mb = measure_memory(MAP_FWD_BKW[mode], x, provider)
            return mem_mb
        raise ValueError(f"bench_kind must be either 'timing' or 'memory', got {bench_kind}")

    result_dfs = benchmark.run(show_plots=False, print_data=False, return_df=True)
    for df, config in zip(result_dfs, configs):
        df.plot(x='N', ylabel = config.ylabel, title=config.plot_name, legend=True)

    return result_dfs


def main():
    print(f"Found {len(KERNELS)} kernels. Thesr kernels are:")
    print("\n".join(x.__name__ for x in KERNELS))

    for kernel in KERNELS:
        print(f"Benchmarking kernel> {kernel.__name__}")

        ref_kernel = import_module(kernel.__module__).torch_ref_module
        do_benchmark(kernel(), ref_kernel())


if __name__ == "__main__":
    main()
