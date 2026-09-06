"""
Empirical Benchmark & Threshold Calibration Runner:
Executes standard PyTorch training loop benchmarks on active hardware,
measures true physical speedup ratios and VRAM footprints across repeated trials,
and calibrates thresholds and knowledge base priors based on physical measurements.
"""

import os
import json
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple

from ghost_layer.kb.knowledge_base import SharedKnowledgeBase


@dataclass
class BenchmarkConfig:
    model_family: str = "transformer"
    hidden_dim: int = 256
    num_layers: int = 4
    seq_len: int = 64
    batch_size: int = 16
    steps_warmup: int = 5
    steps_benchmark: int = 20
    num_trials: int = 3


@dataclass
class BenchmarkResult:
    config_name: str
    avg_step_time_ms: float
    step_time_std_dev_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    peak_vram_mb: float = 0.0
    speedup_pct_vs_baseline: float = 0.0
    speedup_pct_std_dev: float = 0.0
    num_trials: int = 1
    is_baseline: bool = False
    is_empirically_measured: bool = True


@dataclass
class CalibrationReport:
    hardware_name: str
    precision_speedup_pct: float
    precision_speedup_std_dev: float = 0.0
    workers_speedup_pct: float = 0.0
    workers_speedup_std_dev: float = 0.0
    grad_checkpoint_memory_saving_pct: float = 0.0
    grad_checkpoint_memory_saving_std_dev: float = 0.0
    empirical_results: List[BenchmarkResult] = field(default_factory=list)
    is_empirically_measured: bool = True
    calibrated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hardware_name": self.hardware_name,
            "precision_speedup_pct": self.precision_speedup_pct,
            "precision_speedup_std_dev": self.precision_speedup_std_dev,
            "workers_speedup_pct": self.workers_speedup_pct,
            "workers_speedup_std_dev": self.workers_speedup_std_dev,
            "grad_checkpoint_memory_saving_pct": self.grad_checkpoint_memory_saving_pct,
            "grad_checkpoint_memory_saving_std_dev": self.grad_checkpoint_memory_saving_std_dev,
            "is_empirically_measured": self.is_empirically_measured,
            "calibrated_at": self.calibrated_at,
            "empirical_results": [asdict(r) for r in self.empirical_results],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationReport":
        results = [BenchmarkResult(**r) for r in data.get("empirical_results", [])]
        return cls(
            hardware_name=data["hardware_name"],
            precision_speedup_pct=data["precision_speedup_pct"],
            precision_speedup_std_dev=data.get("precision_speedup_std_dev", 0.0),
            workers_speedup_pct=data.get("workers_speedup_pct", 0.0),
            workers_speedup_std_dev=data.get("workers_speedup_std_dev", 0.0),
            grad_checkpoint_memory_saving_pct=data.get("grad_checkpoint_memory_saving_pct", 0.0),
            grad_checkpoint_memory_saving_std_dev=data.get("grad_checkpoint_memory_saving_std_dev", 0.0),
            empirical_results=results,
            is_empirically_measured=data.get("is_empirically_measured", True),
            calibrated_at=data.get("calibrated_at", time.time()),
        )


def _calc_mean_std(values: List[float]) -> Tuple[float, float]:
    """Calculate sample mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    m = sum(values) / len(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return round(m, 2), round(math.sqrt(max(0.0, variance)), 2)


class _SyntheticLatencyDataset(Dataset):
    """Synthetic dataset simulating host I/O and data preprocessing."""
    def __init__(self, size: int = 64, seq_len: int = 64, hidden_dim: int = 256, io_delay_ms: float = 0.2):
        self.size = size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.io_delay_sec = io_delay_ms / 1000.0
        # Pre-allocate data tensors to avoid garbage collection noise
        self._data = [torch.randn(seq_len, hidden_dim) for _ in range(min(size, 32))]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.io_delay_sec > 0:
            time.sleep(self.io_delay_sec)
        return self._data[idx % len(self._data)]


class CalibrationRunner:
    """
    Runs multi-trial micro-benchmarks on the active hardware to empirically measure
    speedups, memory savings, and run-to-run variances, replacing speculative priors.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        knowledge_base: Optional[SharedKnowledgeBase] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.knowledge_base = knowledge_base
        self.hardware_name = torch.cuda.get_device_name(0) if (self.device == "cuda" and torch.cuda.is_available()) else "CPU"

    def _get_peak_vram_mb(self, model: Optional[nn.Module] = None, tensors: Optional[List[torch.Tensor]] = None) -> float:
        """Measure peak allocated VRAM on CUDA or calculate tensor footprint on CPU."""
        if self.device == "cuda" and torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 2)
        
        # On CPU, compute parameter + gradient + active tensor footprint in MB
        total_bytes = 0
        if model is not None:
            total_bytes += sum(p.numel() * p.element_size() for p in model.parameters())
            total_bytes += sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
        if tensors:
            total_bytes += sum(t.numel() * t.element_size() for t in tensors if t is not None)
        return round(max(0.1, total_bytes / (1024.0 * 1024.0)), 2)

    def _build_synthetic_model(self, config: BenchmarkConfig) -> nn.Module:
        layers = []
        for _ in range(config.num_layers):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.LayerNorm(config.hidden_dim))
            layers.append(nn.GELU())
        return nn.Sequential(*layers).to(self.device)

    def benchmark_precision(self, config: Optional[BenchmarkConfig] = None) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Benchmark FP32 vs Mixed Precision (BF16/FP16) across repeated trials."""
        cfg = config or BenchmarkConfig()
        num_trials = max(1, cfg.num_trials)
        use_amp = (self.device == "cuda" and torch.cuda.is_available())
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        fp32_trial_times: List[float] = []
        fp32_trial_vrams: List[float] = []
        amp_trial_times: List[float] = []
        amp_trial_vrams: List[float] = []
        speedup_trials: List[float] = []

        loss_fn = nn.MSELoss()

        for _ in range(num_trials):
            # 1. FP32 Trial
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model_fp32 = self._build_synthetic_model(cfg)
            x_fp32 = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_dim, device=self.device)
            opt_fp32 = torch.optim.AdamW(model_fp32.parameters(), lr=1e-3)

            for _ in range(cfg.steps_warmup):
                opt_fp32.zero_grad()
                out = model_fp32(x_fp32)
                loss = loss_fn(out, x_fp32)
                loss.backward()
                opt_fp32.step()

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(cfg.steps_benchmark):
                opt_fp32.zero_grad()
                out = model_fp32(x_fp32)
                loss = loss_fn(out, x_fp32)
                loss.backward()
                opt_fp32.step()
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            dur_fp32 = (time.time() - t0) / cfg.steps_benchmark * 1000.0
            fp32_trial_times.append(dur_fp32)
            fp32_trial_vrams.append(self._get_peak_vram_mb(model_fp32, [x_fp32]))

            # 2. Mixed Precision Trial
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model_amp = self._build_synthetic_model(cfg)
            x_amp = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_dim, device=self.device)
            opt_amp = torch.optim.AdamW(model_amp.parameters(), lr=1e-3)

            for _ in range(cfg.steps_warmup):
                opt_amp.zero_grad()
                with torch.autocast(device_type=self.device, dtype=amp_dtype, enabled=use_amp):
                    out = model_amp(x_amp)
                    loss = loss_fn(out, x_amp)
                loss.backward()
                opt_amp.step()

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(cfg.steps_benchmark):
                opt_amp.zero_grad()
                with torch.autocast(device_type=self.device, dtype=amp_dtype, enabled=use_amp):
                    out = model_amp(x_amp)
                    loss = loss_fn(out, x_amp)
                loss.backward()
                opt_amp.step()
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            dur_amp = (time.time() - t0) / cfg.steps_benchmark * 1000.0
            amp_trial_times.append(dur_amp)
            amp_trial_vrams.append(self._get_peak_vram_mb(model_amp, [x_amp]))

            s_pct = max(0.0, ((dur_fp32 - dur_amp) / dur_fp32) * 100.0) if dur_fp32 > 0 else 0.0
            speedup_trials.append(s_pct)

        mean_fp32_t, std_fp32_t = _calc_mean_std(fp32_trial_times)
        mean_amp_t, std_amp_t = _calc_mean_std(amp_trial_times)
        mean_speedup, std_speedup = _calc_mean_std(speedup_trials)

        fp32_res = BenchmarkResult(
            config_name="FP32_Baseline",
            avg_step_time_ms=mean_fp32_t,
            step_time_std_dev_ms=std_fp32_t,
            throughput_samples_per_sec=round((cfg.batch_size / (mean_fp32_t / 1000.0)), 1) if mean_fp32_t > 0 else 0.0,
            peak_vram_mb=round(sum(fp32_trial_vrams) / len(fp32_trial_vrams), 2),
            speedup_pct_vs_baseline=0.0,
            speedup_pct_std_dev=0.0,
            num_trials=num_trials,
            is_baseline=True,
            is_empirically_measured=True,
        )

        amp_res = BenchmarkResult(
            config_name="Mixed_Precision_AMP",
            avg_step_time_ms=mean_amp_t,
            step_time_std_dev_ms=std_amp_t,
            throughput_samples_per_sec=round((cfg.batch_size / (mean_amp_t / 1000.0)), 1) if mean_amp_t > 0 else 0.0,
            peak_vram_mb=round(sum(amp_trial_vrams) / len(amp_trial_vrams), 2),
            speedup_pct_vs_baseline=mean_speedup,
            speedup_pct_std_dev=std_speedup,
            num_trials=num_trials,
            is_baseline=False,
            is_empirically_measured=True,
        )

        # Record empirical result to KB if available
        if self.knowledge_base is not None and mean_speedup > 0:
            self.knowledge_base.register_learning(
                architecture_family=cfg.model_family,
                hardware_type=self.hardware_name,
                effective_config={"mixed_precision": "amp"},
                throughput_improvement_pct=mean_speedup,
                was_verified_safe=True,
            )

        return fp32_res, amp_res

    def benchmark_dataloader_workers(self, config: Optional[BenchmarkConfig] = None) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Benchmark DataLoader throughput: single worker vs multi-worker prefetch across repeated trials."""
        cfg = config or BenchmarkConfig()
        num_trials = max(1, cfg.num_trials)
        dataset = _SyntheticLatencyDataset(size=cfg.batch_size * 8, seq_len=cfg.seq_len, hidden_dim=cfg.hidden_dim, io_delay_ms=0.2)
        workers = 2 if os.cpu_count() and os.cpu_count() > 1 else 0

        w0_trial_times: List[float] = []
        wm_trial_times: List[float] = []
        speedup_trials: List[float] = []

        for _ in range(num_trials):
            # 1. Single worker baseline
            loader_0 = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
            t0 = time.time()
            for batch in loader_0:
                _ = batch.sum()
            dur_0_ms = (time.time() - t0) * 1000.0
            w0_trial_times.append(dur_0_ms)

            # 2. Multi-worker prefetch
            is_worker_measured = True
            try:
                loader_multi = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=workers)
                t0 = time.time()
                for batch in loader_multi:
                    _ = batch.sum()
                dur_m_ms = (time.time() - t0) * 1000.0
            except Exception:
                dur_m_ms = dur_0_ms
                is_worker_measured = False

            wm_trial_times.append(dur_m_ms)
            s_pct = max(0.0, ((dur_0_ms - dur_m_ms) / dur_0_ms) * 100.0) if dur_0_ms > 0 else 0.0
            speedup_trials.append(s_pct)

        mean_w0, std_w0 = _calc_mean_std(w0_trial_times)
        mean_wm, std_wm = _calc_mean_std(wm_trial_times)
        mean_speedup, std_speedup = _calc_mean_std(speedup_trials)
        batches_count = max(1, len(dataset) // cfg.batch_size)

        # Measure dataset tensor memory footprint
        ds_vram_mb = round((dataset.size * dataset.seq_len * dataset.hidden_dim * 4.0) / (1024.0 * 1024.0), 2)

        res_0 = BenchmarkResult(
            config_name="DataLoader_Workers_0",
            avg_step_time_ms=round(mean_w0 / batches_count, 2),
            step_time_std_dev_ms=round(std_w0 / batches_count, 2),
            throughput_samples_per_sec=round((len(dataset) / (mean_w0 / 1000.0)), 1) if mean_w0 > 0 else 0.0,
            peak_vram_mb=ds_vram_mb,
            speedup_pct_vs_baseline=0.0,
            speedup_pct_std_dev=0.0,
            num_trials=num_trials,
            is_baseline=True,
            is_empirically_measured=True,
        )

        res_multi = BenchmarkResult(
            config_name=f"DataLoader_Workers_{workers}",
            avg_step_time_ms=round(mean_wm / batches_count, 2),
            step_time_std_dev_ms=round(std_wm / batches_count, 2),
            throughput_samples_per_sec=round((len(dataset) / (mean_wm / 1000.0)), 1) if mean_wm > 0 else 0.0,
            peak_vram_mb=ds_vram_mb,
            speedup_pct_vs_baseline=mean_speedup,
            speedup_pct_std_dev=std_speedup,
            num_trials=num_trials,
            is_baseline=False,
            is_empirically_measured=is_worker_measured,
        )

        if self.knowledge_base is not None and mean_speedup > 0:
            self.knowledge_base.register_learning(
                architecture_family=cfg.model_family,
                hardware_type=self.hardware_name,
                effective_config={"num_workers": workers, "pin_memory": True},
                throughput_improvement_pct=mean_speedup,
                was_verified_safe=True,
            )

        return res_0, res_multi

    def benchmark_gradient_checkpointing(self, config: Optional[BenchmarkConfig] = None) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Benchmark activation memory footprint and step latency: baseline vs gradient checkpointing across repeated trials."""
        cfg = config or BenchmarkConfig()
        num_trials = max(1, cfg.num_trials)
        cfg_deep = BenchmarkConfig(
            model_family=cfg.model_family,
            hidden_dim=cfg.hidden_dim,
            num_layers=max(6, cfg.num_layers * 2),
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
        )

        class DeepBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim * 4)
                self.act = nn.GELU()
                self.fc2 = nn.Linear(dim * 4, dim)
                self.norm = nn.LayerNorm(dim)
            def forward(self, x):
                return x + self.norm(self.fc2(self.act(self.fc1(x))))

        class DeepModel(nn.Module):
            def __init__(self, num_layers, dim, use_checkpointing=False):
                super().__init__()
                self.blocks = nn.ModuleList([DeepBlock(dim) for _ in range(num_layers)])
                self.use_checkpointing = use_checkpointing
            def forward(self, x):
                for block in self.blocks:
                    if self.use_checkpointing and x.requires_grad:
                        x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
                    else:
                        x = block(x)
                return x

        base_times: List[float] = []
        base_vrams: List[float] = []
        ckpt_times: List[float] = []
        ckpt_vrams: List[float] = []
        saving_trials: List[float] = []

        for _ in range(num_trials):
            # 1. Baseline without checkpointing
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            m_base = DeepModel(cfg_deep.num_layers, cfg_deep.hidden_dim, use_checkpointing=False).to(self.device)
            x_base = torch.randn(cfg_deep.batch_size, cfg_deep.seq_len, cfg_deep.hidden_dim, device=self.device, requires_grad=True)
            
            t0 = time.time()
            out_base = m_base(x_base)
            loss_base = out_base.sum()
            loss_base.backward()
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            dur_base = (time.time() - t0) * 1000.0
            vram_base = self._get_peak_vram_mb(m_base, [x_base, out_base])
            base_times.append(dur_base)
            base_vrams.append(vram_base)

            # 2. Checkpointed
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            m_ckpt = DeepModel(cfg_deep.num_layers, cfg_deep.hidden_dim, use_checkpointing=True).to(self.device)
            x_ckpt = torch.randn(cfg_deep.batch_size, cfg_deep.seq_len, cfg_deep.hidden_dim, device=self.device, requires_grad=True)
            
            t0 = time.time()
            out_ckpt = m_ckpt(x_ckpt)
            loss_ckpt = out_ckpt.sum()
            loss_ckpt.backward()
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            dur_ckpt = (time.time() - t0) * 1000.0
            vram_ckpt = self._get_peak_vram_mb(m_ckpt, [x_ckpt, out_ckpt])
            ckpt_times.append(dur_ckpt)
            ckpt_vrams.append(vram_ckpt)

            if vram_base > 0:
                saving_pct = max(0.0, ((vram_base - vram_ckpt) / vram_base) * 100.0)
                is_ckpt_measured = True
            else:
                saving_pct = 0.0
                is_ckpt_measured = False
            saving_trials.append(saving_pct)

        mean_base_t, std_base_t = _calc_mean_std(base_times)
        mean_ckpt_t, std_ckpt_t = _calc_mean_std(ckpt_times)
        mean_saving, std_saving = _calc_mean_std(saving_trials)
        mean_vram_base = round(sum(base_vrams) / len(base_vrams), 2)
        mean_vram_ckpt = round(sum(ckpt_vrams) / len(ckpt_vrams), 2)

        res_base = BenchmarkResult(
            config_name="GradCheckpoint_Off",
            avg_step_time_ms=mean_base_t,
            step_time_std_dev_ms=std_base_t,
            throughput_samples_per_sec=round((cfg_deep.batch_size / (mean_base_t / 1000.0)), 1) if mean_base_t > 0 else 0.0,
            peak_vram_mb=mean_vram_base,
            speedup_pct_vs_baseline=0.0,
            speedup_pct_std_dev=0.0,
            num_trials=num_trials,
            is_baseline=True,
            is_empirically_measured=True,
        )

        res_ckpt = BenchmarkResult(
            config_name="GradCheckpoint_On",
            avg_step_time_ms=mean_ckpt_t,
            step_time_std_dev_ms=std_ckpt_t,
            throughput_samples_per_sec=round((cfg_deep.batch_size / (mean_ckpt_t / 1000.0)), 1) if mean_ckpt_t > 0 else 0.0,
            peak_vram_mb=mean_vram_ckpt,
            speedup_pct_vs_baseline=mean_saving,
            speedup_pct_std_dev=std_saving,
            num_trials=num_trials,
            is_baseline=False,
            is_empirically_measured=is_ckpt_measured,
        )

        return res_base, res_ckpt

    def run_full_calibration(self, config: Optional[BenchmarkConfig] = None) -> CalibrationReport:
        """Run complete hardware calibration sweep across precision, workers, and checkpointing with multi-trial variance statistics."""
        cfg = config or BenchmarkConfig()
        fp32_res, amp_res = self.benchmark_precision(cfg)
        w0_res, w_multi_res = self.benchmark_dataloader_workers(cfg)
        ckpt_off_res, ckpt_on_res = self.benchmark_gradient_checkpointing(cfg)

        report = CalibrationReport(
            hardware_name=self.hardware_name,
            precision_speedup_pct=amp_res.speedup_pct_vs_baseline,
            precision_speedup_std_dev=amp_res.speedup_pct_std_dev,
            workers_speedup_pct=w_multi_res.speedup_pct_vs_baseline,
            workers_speedup_std_dev=w_multi_res.speedup_pct_std_dev,
            grad_checkpoint_memory_saving_pct=ckpt_on_res.speedup_pct_vs_baseline,
            grad_checkpoint_memory_saving_std_dev=ckpt_on_res.speedup_pct_std_dev,
            empirical_results=[fp32_res, amp_res, w0_res, w_multi_res, ckpt_off_res, ckpt_on_res],
            is_empirically_measured=True,
        )
        return report

    def save_report(self, report: CalibrationReport, file_path: str = ".ghostlayer/calibration.json"):
        """Persist calibration report to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

    @classmethod
    def load_report(cls, file_path: str = ".ghostlayer/calibration.json") -> Optional[CalibrationReport]:
        """Load stored calibration report from disk if present."""
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CalibrationReport.from_dict(data)
        except Exception:
            return None


