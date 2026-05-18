import torch
import flash_attn_v100
import csv
import io
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers, tablefmt="grid"):
        del tablefmt
        str_headers = [str(h) for h in headers]
        str_rows = [[str(cell) for cell in row] for row in rows]
        widths = [
            max(len(str_headers[idx]), *(len(row[idx]) for row in str_rows))
            for idx in range(len(str_headers))
        ]
        header_line = " | ".join(str_headers[idx].ljust(widths[idx]) for idx in range(len(str_headers)))
        separator_line = "-+-".join("-" * widths[idx] for idx in range(len(str_headers)))
        body_lines = [
            " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(str_headers)))
            for row in str_rows
        ]
        return "\n".join([header_line, separator_line, *body_lines])

from tensor_parallel import TPDecodeRunner, TensorParallelConfig, benchmark_ms

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    from flash_attn import flash_attn_func as flash_attn_official_func
except ImportError:
    flash_attn_official_func = None


NCU_OCCUPANCY_METRIC = "sm__warps_active.avg.pct_of_peak_sustained_active"


def reference_attention(q, k, v, causal=True):
    """PyTorch 原生 Attention 实现（参考标准）"""
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
    
    if causal:
        n = q.size(-2)
        mask = torch.triu(torch.ones(n, n, device=q.device, dtype=q.dtype), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))
    
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def benchmark_function(func, *args, warmup=10, iterations=100):
    """Benchmark 函数执行时间"""
    # Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()
    
    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        func(*args)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    
    return elapsed_ms / iterations  # 平均时间 (ms)


def calculate_tflops(B, H, N, D, time_ms, _causal=True):
    """
    计算 TFLOPs
    Flash Attention 计算复杂度: O(B * H * N^2 * D)
    - QK^T: B * H * N^2 * D
    - Softmax: 忽略不计
    - PV: B * H * N^2 * D
    总计: 2 * B * H * N^2 * D FLOPs
    """
    flops = 2 * B * H * N * N * D
    time_s = time_ms / 1000.0
    tflops = flops / time_s / 1e12
    return tflops


def calculate_memory_bandwidth(B, H, N, D, time_ms):
    """
    计算内存带宽 (GB/s)
    每次迭代需要读取: Q, K, V, 写入: O
    总计: 4 * B * H * N * D * sizeof(dtype) bytes
    """
    bytes_per_element = 2  # FP16: 2 bytes, FP32: 4 bytes
    total_bytes = 4 * B * H * N * D * bytes_per_element
    time_s = time_ms / 1000.0
    bandwidth_gb_s = total_bytes / time_s / 1e9
    return bandwidth_gb_s


def format_metric(value, digits=3, default="N/A"):
    if value is None:
        return default
    return f"{value:.{digits}f}"


def flash_attn_official_available():
    return flash_attn_official_func is not None


def to_flash_attn_layout(q, k, v):
    return (
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
    )


def from_flash_attn_layout(out):
    return out.transpose(1, 2).contiguous()


def run_flash_attn_official(q, k, v, causal=True):
    if not flash_attn_official_available():
        raise RuntimeError("flash_attn official is not installed")
    q_fa, k_fa, v_fa = to_flash_attn_layout(q, k, v)
    out = flash_attn_official_func(q_fa, k_fa, v_fa, dropout_p=0.0, causal=causal)
    return from_flash_attn_layout(out)


def benchmark_flash_attn_official(q, k, v, causal, warmup, iterations):
    if not flash_attn_official_available():
        return {
            "status": "unavailable",
            "time_ms": None,
            "output": None,
            "error": None,
        }

    try:
        time_ms = benchmark_ms(
            lambda: run_flash_attn_official(q, k, v, causal=causal),
            [q.device],
            warmup=warmup,
            iterations=iterations,
        )
        output = run_flash_attn_official(q, k, v, causal=causal)
        return {
            "status": "ok",
            "time_ms": time_ms,
            "output": output,
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "time_ms": None,
            "output": None,
            "error": str(exc),
        }


def project_root():
    return Path(__file__).resolve().parent


def nsight_compute_path():
    return shutil.which("ncu")


def nsight_systems_path():
    return shutil.which("nsys")


def profiler_script_path():
    return project_root() / "tp_profile_runner.py"


def build_tp_profile_command(config, mode="decode", world_size=None, metric_names=None):
    if len(config) != 6:
        raise ValueError("tp profile config must be a 6-tuple: (B, H_Q, H_KV, N, D, block_size)")
    B, H_Q, H_KV, N, D, block_size = config
    world_size = resolve_tp_world_size(world_size)
    metric_names = metric_names or [NCU_OCCUPANCY_METRIC]

    command = [
        nsight_compute_path() or "ncu",
        "--target-processes",
        "all",
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(metric_names),
        sys.executable,
        str(profiler_script_path()),
        "--mode",
        mode,
        "--world-size",
        str(world_size),
        "--batch-size",
        str(B),
        "--h-q",
        str(H_Q),
        "--h-kv",
        str(H_KV),
        "--seq-len",
        str(N),
        "--head-dim",
        str(D),
        "--block-size",
        str(block_size),
    ]
    return command


def _parse_float_token(value):
    if value is None:
        return None
    cleaned = str(value).strip().replace(",", "")
    if not cleaned or cleaned in {"N/A", "nan", "NaN"}:
        return None
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_ncu_csv_metrics(output_text, metric_names):
    metrics = {metric_name: None for metric_name in metric_names}
    reader = csv.reader(io.StringIO(output_text))
    for row in reader:
        if not row:
            continue
        for metric_name in metric_names:
            if metric_name not in row:
                continue
            metric_index = row.index(metric_name)
            parsed_value = None
            for candidate in reversed(row[metric_index + 1:]):
                parsed_value = _parse_float_token(candidate)
                if parsed_value is not None:
                    break
            if parsed_value is not None:
                metrics[metric_name] = parsed_value
    return metrics


def profile_tp_kernel(config, mode="decode", world_size=None, timeout_s=180):
    world_size = resolve_tp_world_size(world_size)
    result = {
        "status": "disabled",
        "occupancy_pct": None,
        "command": None,
        "stderr": None,
    }

    if not torch.cuda.is_available():
        result["status"] = "cuda_unavailable"
        return result
    if world_size < 2 or torch.cuda.device_count() < world_size:
        result["status"] = "insufficient_gpus"
        return result
    ncu_path = nsight_compute_path()
    if ncu_path is None:
        result["status"] = "ncu_unavailable"
        return result

    command = build_tp_profile_command(
        config,
        mode=mode,
        world_size=world_size,
        metric_names=[NCU_OCCUPANCY_METRIC],
    )
    result["command"] = " ".join(command)

    try:
        completed = subprocess.run(
            command,
            cwd=project_root(),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        return result
    except Exception as exc:
        result["status"] = "launch_error"
        result["stderr"] = str(exc)
        return result

    combined_output = "\n".join(
        part for part in [completed.stdout, completed.stderr] if part
    )
    metrics = parse_ncu_csv_metrics(combined_output, [NCU_OCCUPANCY_METRIC])
    result["occupancy_pct"] = metrics.get(NCU_OCCUPANCY_METRIC)
    if completed.returncode == 0 and result["occupancy_pct"] is not None:
        result["status"] = "ok"
    else:
        result["status"] = f"ncu_error_{completed.returncode}"
        result["stderr"] = completed.stderr.strip()[-500:] if completed.stderr else combined_output[-500:]
    return result


def estimate_attention_io_bandwidth(B, H_Q, H_KV, q_len, kv_len, D, time_ms, bytes_per_element=2):
    if time_ms <= 0:
        return None
    total_elements = B * D * (2 * H_Q * q_len + 2 * H_KV * kv_len)
    total_bytes = total_elements * bytes_per_element
    return total_bytes / (time_ms / 1000.0) / 1e9


def estimate_tp_gather_bytes(B, H_Q, q_len, D, world_size, bytes_per_element=2):
    if world_size <= 1:
        return 0.0
    output_elements = B * H_Q * q_len * D
    return output_elements * bytes_per_element * (world_size - 1) / world_size


class DeviceMetricsMonitor:
    def __init__(self, devices, sample_interval_s=0.05):
        self.sample_interval_s = sample_interval_s
        self.device_indices = []
        for device in devices:
            device = torch.device(device)
            if device.type == "cuda" and device.index is not None:
                self.device_indices.append(device.index)
        self.device_indices = sorted(set(self.device_indices))
        self._gpu_samples = {idx: [] for idx in self.device_indices}
        self._thread = None
        self._running = False
        self._start_time = None
        self._start_nvlink = None
        self._nvml = None
        self._handles = {}

        if not self.device_indices or pynvml is None:
            return

        try:
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = {
                idx: pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in self.device_indices
            }
        except Exception:
            self._nvml = None
            self._handles = {}

    @property
    def available(self):
        return self._nvml is not None and bool(self._handles)

    def _collect_gpu_utilization(self):
        if not self.available:
            return
        for idx, handle in self._handles.items():
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                self._gpu_samples[idx].append(float(util.gpu))
            except Exception:
                continue

    def _counter_value(self, counter, name):
        if counter is None:
            return None
        if hasattr(counter, name):
            return float(getattr(counter, name))
        if isinstance(counter, dict) and name in counter:
            return float(counter[name])
        if isinstance(counter, (tuple, list)) and len(counter) >= 2:
            if name == "rx":
                return float(counter[0])
            if name == "tx":
                return float(counter[1])
        return None

    def _read_nvlink_snapshot(self):
        if not self.available:
            return None
        if not hasattr(self._nvml, "nvmlDeviceGetNvLinkUtilizationCounter"):
            return None

        snapshot = {}
        for idx, handle in self._handles.items():
            total_rx = 0.0
            total_tx = 0.0
            active_links = 0
            for link in range(12):
                try:
                    if hasattr(self._nvml, "nvmlDeviceGetNvLinkState"):
                        state = self._nvml.nvmlDeviceGetNvLinkState(handle, link)
                        if not state:
                            continue
                    counter = self._nvml.nvmlDeviceGetNvLinkUtilizationCounter(handle, link, 0)
                    rx = self._counter_value(counter, "rx")
                    tx = self._counter_value(counter, "tx")
                    if rx is None or tx is None:
                        continue
                    total_rx += rx
                    total_tx += tx
                    active_links += 1
                except Exception:
                    continue
            if active_links > 0:
                snapshot[idx] = {
                    "rx": total_rx,
                    "tx": total_tx,
                    "active_links": active_links,
                }
        return snapshot or None

    def _run(self):
        while self._running:
            self._collect_gpu_utilization()
            time.sleep(self.sample_interval_s)

    def start(self):
        if not self.available or self._running:
            return
        self._collect_gpu_utilization()
        self._start_nvlink = self._read_nvlink_snapshot()
        self._start_time = time.perf_counter()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.available or not self._start_time:
            return {
                "gpu_util_avg": None,
                "nvlink_tx_gbps": None,
                "nvlink_rx_gbps": None,
                "nvlink_util_pct": None,
            }

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self.sample_interval_s * 4))
        self._collect_gpu_utilization()

        elapsed_s = max(time.perf_counter() - self._start_time, 1e-9)
        all_gpu_samples = [
            sample
            for samples in self._gpu_samples.values()
            for sample in samples
        ]
        gpu_util_avg = sum(all_gpu_samples) / len(all_gpu_samples) if all_gpu_samples else None

        end_nvlink = self._read_nvlink_snapshot()
        nvlink_tx_gbps = None
        nvlink_rx_gbps = None
        nvlink_util_pct = None
        if self._start_nvlink and end_nvlink:
            total_tx_bytes = 0.0
            total_rx_bytes = 0.0
            total_peak_gbps = 0.0
            for idx, end_data in end_nvlink.items():
                start_data = self._start_nvlink.get(idx)
                if start_data is None:
                    continue
                total_tx_bytes += max(0.0, end_data["tx"] - start_data["tx"])
                total_rx_bytes += max(0.0, end_data["rx"] - start_data["rx"])
                total_peak_gbps += end_data["active_links"] * 25.0

            nvlink_tx_gbps = total_tx_bytes / elapsed_s / 1e9
            nvlink_rx_gbps = total_rx_bytes / elapsed_s / 1e9
            if total_peak_gbps > 0:
                nvlink_util_pct = ((nvlink_tx_gbps + nvlink_rx_gbps) / (2.0 * total_peak_gbps)) * 100.0

        return {
            "gpu_util_avg": gpu_util_avg,
            "nvlink_tx_gbps": nvlink_tx_gbps,
            "nvlink_rx_gbps": nvlink_rx_gbps,
            "nvlink_util_pct": nvlink_util_pct,
        }


def benchmark_ms_with_observability(fn, devices, warmup=10, iterations=50, sample_interval_s=0.05):
    for _ in range(warmup):
        fn()
    for device in devices:
        torch.cuda.synchronize(device)

    monitor = DeviceMetricsMonitor(devices, sample_interval_s=sample_interval_s)
    monitor.start()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    for device in devices:
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) / iterations * 1000.0
    return elapsed_ms, monitor.stop()


def has_two_gpus():
    return torch.cuda.device_count() >= 2


def resolve_tp_world_size(world_size=None):
    if world_size is not None:
        return world_size
    return torch.cuda.device_count()


def has_required_gpus(world_size):
    return torch.cuda.device_count() >= world_size


def normalize_tp_config(config):
    if len(config) == 5:
        B, H_Q, H_KV, N, D = config
        return B, H_Q, H_KV, N, D, 16
    if len(config) == 6:
        return config
    raise ValueError(f"Unsupported TP config: {config}")


def extract_tp_heads(config):
    if len(config) in (5, 6):
        _, H_Q, H_KV, _, _, _ = normalize_tp_config(config)
        return H_Q, H_KV
    if len(config) == 7:
        _, H_Q, H_KV, _, _, _, _ = config
        return H_Q, H_KV
    raise ValueError(f"Unsupported TP config: {config}")


def is_tp_compatible_config(config, world_size):
    H_Q, H_KV = extract_tp_heads(config)
    return H_Q % world_size == 0 and H_KV % world_size == 0


def filter_compatible_tp_configs(configs, world_size):
    return [config for config in configs if is_tp_compatible_config(config, world_size)]


def default_tp_microbenchmark_configs(world_size=2):
    candidate_configs = [
        (1, 16, 4, 1024, 64, 16),
        (1, 16, 4, 1024, 64, 32),
        (1, 16, 1, 1024, 64, 64),
        (2, 16, 4, 1024, 64, 16),
        (1, 32, 8, 2048, 128, 16),
        (1, 32, 4, 2048, 128, 16),
        (1, 32, 1, 2048, 128, 16),
        (1, 32, 8, 2048, 128, 32),
        (1, 32, 4, 2048, 128, 64),
        (2, 32, 8, 2048, 128, 16),
        (1, 24, 6, 1024, 64, 16),
        (1, 24, 6, 1024, 64, 32),
        (1, 24, 3, 1024, 64, 64),
        (2, 24, 6, 1024, 64, 16),
        (1, 48, 12, 2048, 128, 16),
        (1, 48, 6, 2048, 128, 32),
        (1, 48, 3, 2048, 128, 64),
        (2, 48, 12, 2048, 128, 16),
    ]
    configs = filter_compatible_tp_configs(candidate_configs, world_size)
    if not configs:
        raise ValueError(f"No default TP microbenchmark configs for world_size={world_size}")
    return configs


def default_tp_end_to_end_configs(world_size=2):
    candidate_configs = [
        (1, 16, 4, 256, 64, 32, 16),
        (1, 16, 4, 256, 64, 32, 32),
        (1, 32, 8, 512, 128, 32, 16),
        (2, 32, 8, 512, 128, 32, 16),
        (1, 32, 4, 512, 128, 32, 32),
        (1, 32, 1, 512, 128, 32, 64),
        (1, 24, 6, 256, 64, 32, 16),
        (1, 24, 6, 256, 64, 32, 32),
        (1, 48, 12, 512, 128, 32, 16),
        (2, 48, 12, 512, 128, 32, 16),
        (1, 48, 6, 512, 128, 32, 32),
        (1, 48, 3, 512, 128, 32, 64),
    ]
    configs = filter_compatible_tp_configs(candidate_configs, world_size)
    if not configs:
        raise ValueError(f"No default TP end-to-end configs for world_size={world_size}")
    return configs


def run_tp_benchmark(
    configs=None,
    world_size=None,
    prefill_warmup=5,
    prefill_iterations=20,
    decode_warmup=10,
    decode_iterations=50,
    profile_occupancy=False,
):
    """运行 Tensor Parallel benchmark"""
    world_size = resolve_tp_world_size(world_size)
    print("\n" + "=" * 100)
    print(f"Tensor Parallel V100 Benchmark (TP={world_size})")
    print("=" * 100)

    if world_size < 2 or not has_required_gpus(world_size):
        print(f"Skip TP benchmark: need at least {max(2, world_size)} GPUs")
        return []

    configs = configs or default_tp_microbenchmark_configs(world_size=world_size)
    configs = filter_compatible_tp_configs(configs, world_size)
    if not configs:
        print(f"Skip TP benchmark: no configs compatible with world_size={world_size}")
        return []

    results = []

    for raw_config in configs:
        B, H_Q, H_KV, N, D, block_size = normalize_tp_config(raw_config)
        paged_supported = block_size == 16
        runner = TPDecodeRunner(
            TensorParallelConfig(
                H_Q=H_Q,
                H_KV=H_KV,
                D=D,
                world_size=world_size,
                causal=True,
                auto_grow_kv_cache=True,
                kv_cache_page_block_size=block_size,
                kv_cache_layout="blocked",
            )
        )

        q_prefill = torch.randn(B, H_Q, N, D, device="cuda:0", dtype=torch.float16).contiguous()
        k_prefill = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16).contiguous()
        v_prefill = torch.randn(B, H_KV, N, D, device="cuda:0", dtype=torch.float16).contiguous()

        q_decode = torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16).contiguous()
        k_decode = k_prefill
        v_decode = v_prefill

        prefill_time = benchmark_ms(
            lambda: runner.prefill(q_prefill, k_prefill, v_prefill, use_workspace=True),
            runner.decoder.devices,
            warmup=prefill_warmup,
            iterations=prefill_iterations,
        )
        decode_time = benchmark_ms(
            lambda: runner.decode(q_decode, k_decode, v_decode, cache_len=N - 1, use_workspace=True),
            runner.decoder.devices,
            warmup=decode_warmup,
            iterations=decode_iterations,
        )
        kv_cache = runner.create_kv_cache(B, N, layout="blocked")
        runner.prefill_to_kv_cache(q_prefill, k_prefill, v_prefill, kv_cache, return_output=False, use_workspace=True)
        if paged_supported:
            paged_decode_time = benchmark_ms(
                lambda: runner.decode_with_paged_kv_cache(q_decode, kv_cache, use_workspace=True),
                runner.decoder.devices,
                warmup=decode_warmup,
                iterations=decode_iterations,
            )
        else:
            paged_decode_time = None

        official_prefill = benchmark_flash_attn_official(
            q_prefill,
            k_prefill,
            v_prefill,
            causal=True,
            warmup=prefill_warmup,
            iterations=prefill_iterations,
        )
        official_decode = benchmark_flash_attn_official(
            q_decode,
            k_decode,
            v_decode,
            causal=True,
            warmup=decode_warmup,
            iterations=decode_iterations,
        )

        out_tp_decode = runner.decode(q_decode, k_decode, v_decode, cache_len=N - 1, use_workspace=True)
        out_ref_decode = flash_attn_v100.forward_decode_gqa_fp16(q_decode, k_decode, v_decode, True, N - 1)
        decode_diff = (out_tp_decode.float() - out_ref_decode.float()).abs().max().item()
        official_decode_diff = (
            (out_tp_decode.float() - official_decode["output"].float()).abs().max().item()
            if official_decode["output"] is not None
            else None
        )
        if paged_supported:
            out_tp_paged = runner.decode_with_paged_kv_cache(q_decode, kv_cache, use_workspace=True)
            paged_decode_diff = (out_tp_paged.float() - out_ref_decode.float()).abs().max().item()
        else:
            paged_decode_diff = None

        out_tp_prefill = runner.prefill(q_prefill, k_prefill, v_prefill, use_workspace=True)
        out_ref_prefill = flash_attn_v100.forward_prefill_gqa_fp16(q_prefill, k_prefill, v_prefill, True)
        prefill_diff = (out_tp_prefill.float() - out_ref_prefill.float()).abs().max().item()
        official_prefill_diff = (
            (out_tp_prefill.float() - official_prefill["output"].float()).abs().max().item()
            if official_prefill["output"] is not None
            else None
        )

        cache_meta = runner.get_kv_cache_metadata(kv_cache)
        block_meta = runner.get_kv_cache_block_metadata(kv_cache)
        stats = runner.get_runtime_stats()

        prefill_tokens_per_s = (B * N) / (prefill_time / 1000.0)
        decode_tokens_per_s = B / (decode_time / 1000.0)
        paged_decode_tokens_per_s = (B / (paged_decode_time / 1000.0)) if paged_supported else None
        gqa_ratio = H_Q // H_KV
        prefill_latency_us = prefill_time * 1000.0 / (B * N)
        decode_latency_us = decode_time * 1000.0 / B
        prefill_tflops = calculate_tflops(B, H_Q, N, D, prefill_time)
        prefill_hbm_bw = estimate_attention_io_bandwidth(B, H_Q, H_KV, N, N, D, prefill_time)
        decode_hbm_bw = estimate_attention_io_bandwidth(B, H_Q, H_KV, 1, N, D, decode_time)
        sweep_tag = f"b{B}-d{D}-n{N}-gqa{gqa_ratio}-blk{block_size}"
        normalized_config = (B, H_Q, H_KV, N, D, block_size)
        official_prefill_speedup = (
            official_prefill["time_ms"] / prefill_time
            if official_prefill["time_ms"] is not None and prefill_time > 0
            else None
        )
        official_decode_speedup = (
            official_decode["time_ms"] / decode_time
            if official_decode["time_ms"] is not None and decode_time > 0
            else None
        )
        if profile_occupancy:
            prefill_profile = profile_tp_kernel(normalized_config, mode="prefill", world_size=world_size)
            decode_profile = profile_tp_kernel(normalized_config, mode="decode", world_size=world_size)
        else:
            prefill_profile = {"status": "disabled", "occupancy_pct": None}
            decode_profile = {"status": "disabled", "occupancy_pct": None}

        results.append({
            "B": B,
            "TP": world_size,
            "H_Q": H_Q,
            "H_KV": H_KV,
            "GQA ratio": gqa_ratio,
            "N": N,
            "D": D,
            "Sweep tag": sweep_tag,
            "Prefill (ms)": f"{prefill_time:.3f}",
            "Prefill lat/token (us)": f"{prefill_latency_us:.3f}",
            "Prefill tok/s": f"{prefill_tokens_per_s:.1f}",
            "Prefill TFLOPS": f"{prefill_tflops:.2f}",
            "Prefill HBM (GB/s)": format_metric(prefill_hbm_bw, digits=2),
            "Prefill occ (%)": format_metric(prefill_profile["occupancy_pct"], digits=1),
            "Prefill diff": f"{prefill_diff:.6f}",
            "Official prefill (ms)": format_metric(official_prefill["time_ms"], digits=3),
            "Official prefill diff": format_metric(official_prefill_diff, digits=6),
            "Official prefill speedup": format_metric(official_prefill_speedup, digits=2),
            "Decode (ms)": f"{decode_time:.3f}",
            "Decode lat/token (us)": f"{decode_latency_us:.3f}",
            "Decode tok/s": f"{decode_tokens_per_s:.1f}",
            "Decode HBM (GB/s)": format_metric(decode_hbm_bw, digits=2),
            "Decode occ (%)": format_metric(decode_profile["occupancy_pct"], digits=1),
            "Decode diff": f"{decode_diff:.6f}",
            "Official decode (ms)": format_metric(official_decode["time_ms"], digits=3),
            "Official decode diff": format_metric(official_decode_diff, digits=6),
            "Official decode speedup": format_metric(official_decode_speedup, digits=2),
            "Paged Decode (ms)": f"{paged_decode_time:.3f}" if paged_supported else "N/A",
            "Paged tok/s": f"{paged_decode_tokens_per_s:.1f}" if paged_supported else "N/A",
            "Paged diff": f"{paged_decode_diff:.6f}" if paged_supported else "N/A",
            "KV layout": cache_meta["layout"],
            "Block size": block_meta["tokens_per_block"],
            "Blocks": block_meta["allocated_block_count"],
            "Paged path": "native" if paged_supported else "unsupported",
            "Official baseline": official_decode["status"] if official_decode["status"] != "unavailable" else official_prefill["status"],
            "Occ profiler": decode_profile["status"] if decode_profile["status"] != "disabled" else prefill_profile["status"],
            "Grow cnt": cache_meta["grow_count"],
            "Last op": stats["last_operation"],
        })

        print(
            f"  TP={world_size}, B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D} | "
            f"prefill={prefill_time:.3f}ms ({prefill_tokens_per_s:.1f} tok/s), "
            f"decode={decode_time:.3f}ms ({decode_tokens_per_s:.1f} tok/s), "
            f"paged={'N/A' if not paged_supported else f'{paged_decode_time:.3f}ms ({paged_decode_tokens_per_s:.1f} tok/s)'}, "
            f"ratio={gqa_ratio}, block_size={block_size}, layout={cache_meta['layout']}, blocks={block_meta['allocated_block_count']}, "
            f"official={official_decode['status']} speedup={format_metric(official_decode_speedup, digits=2)}, "
            f"occ={format_metric(decode_profile['occupancy_pct'], digits=1)} ({decode_profile['status']})"
        )

    return results


def run_tp_end_to_end_benchmark(configs=None, world_size=None, prefill_warmup=2, prefill_iterations=5, decode_warmup=1, decode_iterations=1):
    """运行 TP 端到端 benchmark，记录 TTFT / ITL / tok/s"""
    world_size = resolve_tp_world_size(world_size)
    print("\n" + "=" * 100)
    print(f"Tensor Parallel End-to-End Benchmark (TP={world_size})")
    print("=" * 100)

    if world_size < 2 or not has_required_gpus(world_size):
        print(f"Skip TP end-to-end benchmark: need at least {max(2, world_size)} GPUs")
        return []

    configs = configs or default_tp_end_to_end_configs(world_size=world_size)
    configs = filter_compatible_tp_configs(configs, world_size)
    if not configs:
        print(f"Skip TP end-to-end benchmark: no configs compatible with world_size={world_size}")
        return []

    results = []

    for B, H_Q, H_KV, prompt_len, D, new_tokens, block_size in configs:
        paged_supported = block_size == 16
        runner = TPDecodeRunner(
            TensorParallelConfig(
                H_Q=H_Q,
                H_KV=H_KV,
                D=D,
                world_size=world_size,
                causal=True,
                auto_grow_kv_cache=True,
                kv_cache_page_block_size=block_size,
                kv_cache_layout="blocked",
            )
        )
        kv_cache = runner.create_kv_cache(B, prompt_len + new_tokens + block_size, layout="blocked")

        q_prompt = torch.randn(B, H_Q, prompt_len, D, device="cuda:0", dtype=torch.float16).contiguous()
        k_prompt = torch.randn(B, H_KV, prompt_len, D, device="cuda:0", dtype=torch.float16).contiguous()
        v_prompt = torch.randn(B, H_KV, prompt_len, D, device="cuda:0", dtype=torch.float16).contiguous()

        ttft_prefill_ms = benchmark_ms(
            lambda: runner.prefill_to_kv_cache(q_prompt, k_prompt, v_prompt, kv_cache, return_output=False, use_workspace=True),
            runner.decoder.devices,
            warmup=prefill_warmup,
            iterations=prefill_iterations,
        )

        q_steps = [torch.randn(B, H_Q, 1, D, device="cuda:0", dtype=torch.float16).contiguous() for _ in range(new_tokens)]
        k_steps = [torch.randn(B, H_KV, 1, D, device="cuda:0", dtype=torch.float16).contiguous() for _ in range(new_tokens)]
        v_steps = [torch.randn(B, H_KV, 1, D, device="cuda:0", dtype=torch.float16).contiguous() for _ in range(new_tokens)]

        def run_first_token():
            runner.reset_kv_cache(kv_cache)
            runner.prefill_to_kv_cache(q_prompt, k_prompt, v_prompt, kv_cache, return_output=False, use_workspace=True)
            if paged_supported:
                runner.decode_with_paged_kv_cache(q_steps[0], kv_cache, new_k=k_steps[0], new_v=v_steps[0], use_workspace=True)
            else:
                runner.decode_with_kv_cache(q_steps[0], kv_cache, new_k=k_steps[0], new_v=v_steps[0], use_workspace=True)

        ttft_total_ms = benchmark_ms(
            run_first_token,
            runner.decoder.devices,
            warmup=decode_warmup,
            iterations=max(1, decode_iterations),
        )

        def run_decode_sequence():
            runner.reset_kv_cache(kv_cache)
            runner.prefill_to_kv_cache(q_prompt, k_prompt, v_prompt, kv_cache, return_output=False, use_workspace=True)
            for step in range(new_tokens):
                if paged_supported:
                    runner.decode_with_paged_kv_cache(
                        q_steps[step],
                        kv_cache,
                        new_k=k_steps[step],
                        new_v=v_steps[step],
                        use_workspace=True,
                    )
                else:
                    runner.decode_with_kv_cache(
                        q_steps[step],
                        kv_cache,
                        new_k=k_steps[step],
                        new_v=v_steps[step],
                        use_workspace=True,
                    )

        decode_sequence_ms, e2e_metrics = benchmark_ms_with_observability(
            run_decode_sequence,
            runner.decoder.devices,
            warmup=decode_warmup,
            iterations=max(1, decode_iterations),
        )

        runner.reset_kv_cache(kv_cache)
        runner.prefill_to_kv_cache(q_prompt, k_prompt, v_prompt, kv_cache, return_output=False, use_workspace=True)
        per_token_ms = []
        for step in range(new_tokens):
            if paged_supported:
                decode_callable = lambda step=step: runner.decode_with_paged_kv_cache(
                    q_steps[step],
                    kv_cache,
                    new_k=k_steps[step],
                    new_v=v_steps[step],
                    use_workspace=True,
                )
            else:
                decode_callable = lambda step=step: runner.decode_with_kv_cache(
                    q_steps[step],
                    kv_cache,
                    new_k=k_steps[step],
                    new_v=v_steps[step],
                    use_workspace=True,
                )
            token_ms = benchmark_ms(
                decode_callable,
                runner.decoder.devices,
                warmup=0,
                iterations=1,
            )
            per_token_ms.append(token_ms)

        total_decode_ms = decode_sequence_ms
        itl_ms = (sum(per_token_ms[1:]) / max(1, len(per_token_ms) - 1)) if len(per_token_ms) > 1 else per_token_ms[0]
        tok_per_s = (B * new_tokens) / (total_decode_ms / 1000.0)
        total_gather_bytes = estimate_tp_gather_bytes(
            B, H_Q, prompt_len, D, runner.config.world_size
        ) + new_tokens * estimate_tp_gather_bytes(
            B, H_Q, 1, D, runner.config.world_size
        )
        nvlink_est_gbps = total_gather_bytes / (total_decode_ms / 1000.0) / 1e9 if total_decode_ms > 0 else None

        results.append({
            "B": B,
            "TP": world_size,
            "H_Q": H_Q,
            "H_KV": H_KV,
            "GQA ratio": H_Q // H_KV,
            "Prompt len": prompt_len,
            "New tokens": new_tokens,
            "D": D,
            "Block size": block_size,
            "Decode path": "paged" if paged_supported else "contiguous",
            "TTFT prefill (ms)": f"{ttft_prefill_ms:.3f}",
            "TTFT total (ms)": f"{ttft_total_ms:.3f}",
            "ITL (ms)": f"{itl_ms:.3f}",
            "tok/s": f"{tok_per_s:.1f}",
            "GPU util (%)": format_metric(e2e_metrics["gpu_util_avg"], digits=1),
            "NVLink TX (GB/s)": format_metric(e2e_metrics["nvlink_tx_gbps"], digits=2),
            "NVLink RX (GB/s)": format_metric(e2e_metrics["nvlink_rx_gbps"], digits=2),
            "NVLink util (%)": format_metric(e2e_metrics["nvlink_util_pct"], digits=1),
            "NVLink est. (GB/s)": format_metric(nvlink_est_gbps, digits=2),
        })

        print(
            f"  TP={world_size}, B={B}, H_Q={H_Q}, H_KV={H_KV}, prompt={prompt_len}, new={new_tokens}, D={D}, block={block_size} | "
            f"TTFT_prefill={ttft_prefill_ms:.3f}ms, TTFT_total={ttft_total_ms:.3f}ms, ITL={itl_ms:.3f}ms, tok/s={tok_per_s:.1f}, "
            f"gpu_util={format_metric(e2e_metrics['gpu_util_avg'], digits=1)}, "
            f"nvlink_est={format_metric(nvlink_est_gbps, digits=2)}, "
            f"path={'paged' if paged_supported else 'contiguous'}"
        )

    return results


def run_benchmark():
    """运行完整的 benchmark 测试"""
    print("=" * 100)
    print("Flash Attention V100 Benchmark")
    print("=" * 100)
    
    # 测试配置
    configs = [
        # (B, H, N, D)
        (1, 8, 128, 64),
        (1, 8, 256, 64),
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
        (2, 8, 256, 64),
        (2, 8, 512, 64),
        (2, 8, 1024, 64),
        (2, 8, 2048, 64),
        (1, 8, 256, 128),
        (1, 8, 512, 128),
        (1, 8, 1024, 128),
        (2, 8, 256, 128),
        (2, 8, 512, 128),
        (2, 8, 1024, 128),
    ]
    
    results = []
    
    for B, H, N, D in configs:
        print(f"\nTesting: B={B}, H={H}, N={N}, D={D}")
        print("-" * 100)
        
        # 创建测试数据 (FP16)
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        
        # 确保连续性
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        for causal in [False, True]:
            causal_str = "Causal" if causal else "Non-Causal"
            
            # Benchmark Flash Attention FP16
            try:
                time_fa = benchmark_function(
                    flash_attn_v100.forward_fp16,
                    q, k, v, causal,
                    warmup=10,
                    iterations=100
                )
                
                # 验证正确性
                out_fa = flash_attn_v100.forward_fp16(q, k, v, causal)
                out_ref = reference_attention(q, k, v, causal)
                
                max_diff = (out_fa.float() - out_ref.float()).abs().max().item()
                all_close = torch.allclose(out_fa, out_ref, atol=1e-2, rtol=1e-2)
                
                # 计算性能指标
                tflops = calculate_tflops(B, H, N, D, time_fa, causal)
                bandwidth = calculate_memory_bandwidth(B, H, N, D, time_fa)
                
                results.append({
                    'B': B, 'H': H, 'N': N, 'D': D,
                    'Mode': causal_str,
                    'Time (ms)': f"{time_fa:.3f}",
                    'TFLOPs': f"{tflops:.2f}",
                    'BW (GB/s)': f"{bandwidth:.2f}",
                    'Max Diff': f"{max_diff:.6f}",
                    'Correct': '✓' if all_close else '✗'
                })
                
                print(f"  {causal_str:12s} | Time: {time_fa:.3f} ms | "
                      f"TFLOPs: {tflops:.2f} | BW: {bandwidth:.2f} GB/s | "
                      f"Max Diff: {max_diff:.6f} | Correct: {all_close}")
                
            except Exception as e:
                print(f"  {causal_str:12s} | ERROR: {str(e)}")
                results.append({
                    'B': B, 'H': H, 'N': N, 'D': D,
                    'Mode': causal_str,
                    'Time (ms)': 'ERROR',
                    'TFLOPs': 'N/A',
                    'BW (GB/s)': 'N/A',
                    'Max Diff': 'N/A',
                    'Correct': '✗'
                })
        
        torch.cuda.empty_cache()
    
    # 打印汇总表
    print("\n" + "=" * 100)
    print("Benchmark Summary")
    print("=" * 100)
    
    headers = ['B', 'H', 'N', 'D', 'Mode', 'Time (ms)', 'TFLOPs', 'BW (GB/s)', 'Max Diff', 'Correct']
    table_data = []
    
    for r in results:
        table_data.append([
            r['B'], r['H'], r['N'], r['D'], r['Mode'],
            r['Time (ms)'], r['TFLOPs'], r['BW (GB/s)'],
            r['Max Diff'], r['Correct']
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # 保存结果到文件
    with open('benchmark_results.txt', 'w') as f:
        f.write("Flash Attention V100 Benchmark Results\n")
        f.write("=" * 100 + "\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print("\nResults saved to benchmark_results.txt")
    
    # Roofline 分析提示
    print("\n" + "=" * 100)
    print("Roofline Analysis Notes")
    print("=" * 100)
    print("""
V100 GPU 理论峰值:
  - FP16 Tensor Core: ~120 TFLOPs (with mixed precision)
  - FP16 CUDA Core: ~15.7 TFLOPs
  - Memory Bandwidth: 900 GB/s (HBM2)

性能优化建议:
  1. 如果 TFLOPs 较低 → 计算密集型，需要优化计算（如使用 Tensor Core）
  2. 如果带宽接近 900 GB/s → 内存密集型，需要优化内存访问
  3. 当前实现使用 FP16 + 向量化，应能充分利用内存带宽
  4. 要进一步优化可考虑:
     - 使用 WMMA (Warp-level Matrix Multiply-Accumulate) 指令
     - 使用 Tensor Core (需要特殊的矩阵布局)
     - 优化 shared memory bank conflicts
    """)

    tp_results = run_tp_benchmark()
    if tp_results:
        print("\n" + "=" * 100)
        print("Tensor Parallel Benchmark Summary")
        print("=" * 100)
        headers = [
            "TP", "B", "H_Q", "H_KV", "GQA ratio", "N", "D", "Sweep tag",
            "Prefill (ms)", "Prefill lat/token (us)", "Prefill tok/s", "Prefill TFLOPS", "Prefill HBM (GB/s)", "Prefill occ (%)", "Prefill diff",
            "Official prefill (ms)", "Official prefill diff", "Official prefill speedup",
            "Decode (ms)", "Decode lat/token (us)", "Decode tok/s", "Decode HBM (GB/s)", "Decode occ (%)", "Decode diff",
            "Official decode (ms)", "Official decode diff", "Official decode speedup",
            "Paged Decode (ms)", "Paged tok/s", "Paged diff",
            "KV layout", "Block size", "Blocks", "Paged path", "Official baseline", "Occ profiler", "Grow cnt", "Last op",
        ]
        table_data = [[r[h] for h in headers] for r in tp_results]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        with open("benchmark_results.txt", "a") as f:
            f.write("\n\nTensor Parallel Benchmark Results\n")
            f.write("=" * 100 + "\n\n")
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))

    tp_e2e_results = run_tp_end_to_end_benchmark()
    if tp_e2e_results:
        print("\n" + "=" * 100)
        print("Tensor Parallel End-to-End Summary")
        print("=" * 100)
        headers = [
            "TP", "B", "H_Q", "H_KV", "GQA ratio", "Prompt len", "New tokens", "D",
            "Block size", "Decode path", "TTFT prefill (ms)", "TTFT total (ms)", "ITL (ms)", "tok/s",
            "GPU util (%)", "NVLink TX (GB/s)", "NVLink RX (GB/s)", "NVLink util (%)", "NVLink est. (GB/s)",
        ]
        table_data = [[r[h] for h in headers] for r in tp_e2e_results]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        with open("benchmark_results.txt", "a") as f:
            f.write("\n\nTensor Parallel End-to-End Benchmark Results\n")
            f.write("=" * 100 + "\n\n")
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == '__main__':
    run_benchmark()
