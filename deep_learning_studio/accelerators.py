from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AcceleratorInfo:
    key: str
    label: str
    available: bool
    device: torch.device | None
    detail: str


def _directml_device() -> tuple[torch.device | None, str]:
    if importlib.util.find_spec("torch_directml") is None:
        return None, "torch-directml 未導入"
    try:
        import torch_directml

        device = torch_directml.device()
        # A real operation catches broken driver/runtime combinations.
        result = (torch.ones(1).to(device) + 1).cpu().item()
        if result != 2:
            raise RuntimeError("DirectML verification returned an unexpected result")
        name = (
            torch_directml.device_name(0)
            if hasattr(torch_directml, "device_name")
            else "DirectML device"
        )
        return device, str(name)
    except Exception as error:
        return None, f"初期化失敗: {error}"


def detect_accelerators() -> list[AcceleratorInfo]:
    accelerators: list[AcceleratorInfo] = []
    hip_version = getattr(torch.version, "hip", None)
    cuda_build = getattr(torch.version, "cuda", None)

    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    if hip_version:
        detail = (
            f"{torch.cuda.get_device_name(0)} / ROCm {hip_version}"
            if cuda_available
            else f"ROCm {hip_version} buildですがGPUを初期化できません"
        )
        accelerators.append(
            AcceleratorInfo(
                "rocm",
                "ROCm (AMD GPU)",
                cuda_available,
                torch.device("cuda") if cuda_available else None,
                detail,
            )
        )
        accelerators.append(
            AcceleratorInfo(
                "cuda", "CUDA (NVIDIA GPU)", False, None, "ROCm版PyTorchが読み込まれています"
            )
        )
    else:
        if cuda_available:
            properties = torch.cuda.get_device_properties(0)
            memory_gib = properties.total_memory / 1024**3
            detail = f"{properties.name} / {memory_gib:.1f} GiB / PyTorch CUDA {cuda_build}"
        elif cuda_build is None:
            detail = "現在のPyTorchはCUDAを含まないCPU版です"
        else:
            detail = f"PyTorch CUDA {cuda_build} buildですがGPUを初期化できません"
        accelerators.append(
            AcceleratorInfo(
                "cuda",
                "CUDA (NVIDIA GPU)",
                cuda_available,
                torch.device("cuda") if cuda_available else None,
                detail,
            )
        )
        accelerators.append(
            AcceleratorInfo("rocm", "ROCm (AMD GPU)", False, None, "ROCm版PyTorchではありません")
        )

    xpu_api = getattr(torch, "xpu", None)
    try:
        xpu_available = bool(xpu_api and xpu_api.is_available())
    except Exception:
        xpu_available = False
    xpu_detail = (
        str(xpu_api.get_device_name(0))
        if xpu_available and hasattr(xpu_api, "get_device_name")
        else "Intel XPU版PyTorchまたは対応ドライバーが必要です"
    )
    accelerators.append(
        AcceleratorInfo(
            "xpu",
            "XPU (Intel GPU)",
            xpu_available,
            torch.device("xpu") if xpu_available else None,
            xpu_detail,
        )
    )

    mps_api = getattr(torch.backends, "mps", None)
    try:
        mps_available = bool(mps_api and mps_api.is_available())
    except Exception:
        mps_available = False
    accelerators.append(
        AcceleratorInfo(
            "mps",
            "MPS (Apple Silicon GPU)",
            mps_available,
            torch.device("mps") if mps_available else None,
            "Apple Metal Performance Shaders"
            if mps_available
            else "Apple Silicon環境でのみ利用できます",
        )
    )

    directml, directml_detail = _directml_device()
    accelerators.append(
        AcceleratorInfo(
            "directml", "DirectML (Windows GPU)", directml is not None, directml, directml_detail
        )
    )
    accelerators.append(
        AcceleratorInfo(
            "cpu",
            "CPU",
            True,
            torch.device("cpu"),
            f"{platform.processor() or 'CPU'} / PyTorch threads: {torch.get_num_threads()}",
        )
    )
    return accelerators


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    accelerators = detect_accelerators()
    if requested == "auto":
        for key in ("cuda", "rocm", "xpu", "mps", "directml", "cpu"):
            item = next(candidate for candidate in accelerators if candidate.key == key)
            if item.available and item.device is not None:
                return item.device
    item = next((candidate for candidate in accelerators if candidate.key == requested), None)
    if item is None:
        # Keep explicit device strings useful for advanced/multi-GPU users.
        if ":" in requested:
            return torch.device(requested)
        raise ValueError(f"Unknown accelerator: {requested}")
    if not item.available or item.device is None:
        raise RuntimeError(f"{item.label} は利用できません: {item.detail}")
    return item.device


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        name = torch.cuda.get_device_name(device)
        memory_gib = torch.cuda.get_device_properties(device).total_memory / 1024**3
        return f"{backend}: {name} ({memory_gib:.1f} GiB)"
    if device.type == "xpu":
        name = (
            torch.xpu.get_device_name(device)
            if hasattr(torch.xpu, "get_device_name")
            else "Intel GPU"
        )
        return f"XPU: {name}"
    if device.type == "mps":
        return "Apple Metal (MPS)"
    if device.type == "privateuseone":
        return "Windows GPU (DirectML)"
    return f"CPU ({torch.get_num_threads()} threads)"


def accelerator_report() -> str:
    lines = [
        "Deep Learning Studio — Accelerator Diagnostics",
        f"OS: {platform.platform()}",
        f"Python: {sys.version.split()[0]} ({sys.executable})",
        f"PyTorch: {torch.__version__}",
        f"PyTorch CUDA runtime: {getattr(torch.version, 'cuda', None) or 'なし（CPU版）'}",
        f"PyTorch ROCm runtime: {getattr(torch.version, 'hip', None) or 'なし'}",
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未設定')}",
        "",
        "検出されたバックエンド:",
    ]
    for item in detect_accelerators():
        status = "利用可能" if item.available else "利用不可"
        lines.append(f"- {item.label}: {status} — {item.detail}")

    lines.extend(["", "nvidia-smi:"])
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        output = completed.stdout.strip() or completed.stderr.strip()
        lines.append(output or f"終了コード {completed.returncode}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        lines.append(f"実行できません: {error}")

    if getattr(torch.version, "cuda", None) is None:
        lines.extend(
            [
                "",
                "推定原因: CUDAを含まないPyTorchがインストールされています。",
                "NVIDIA GPUがあってもCPU版PyTorchでは torch.cuda.is_available() はFalseになります。",
                "Windowsでは scripts/install_windows_cuda.ps1 を実行してから再起動してください。",
            ]
        )
    return "\n".join(lines)
