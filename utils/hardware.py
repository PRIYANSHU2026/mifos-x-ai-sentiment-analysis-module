import platform
import psutil
import torch
import shutil

try:
    import GPUtil
except ImportError:
    GPUtil = None

def get_size(bytes_val, suffix="B"):
    """Scale bytes to its proper format."""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes_val < factor:
            return f"{bytes_val:.2f}{unit}{suffix}"
        bytes_val /= factor
    return f"{bytes_val:.2f}Y{suffix}"

def detect_hardware():
    info = {
        "os": platform.system(),
        "os_version": platform.release(),
        "cpu_brand": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "cpu_utilization": psutil.cpu_percent(interval=0.1),
        "ram_total": get_size(psutil.virtual_memory().total),
        "ram_available": get_size(psutil.virtual_memory().available),
        "ram_percent": psutil.virtual_memory().percent,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        "gpus": []
    }

    # GPU Detection
    if info["cuda_available"] and GPUtil:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            info["gpus"].append({
                "name": gpu.name,
                "load": f"{gpu.load*100:.1f}%",
                "free_mem": f"{gpu.memoryFree}MB",
                "used_mem": f"{gpu.memoryUsed}MB",
                "total_mem": f"{gpu.memoryTotal}MB",
                "temperature": f"{gpu.temperature} °C"
            })
    elif info["mps_available"]:
        info["gpus"].append({
            "name": "Apple Silicon GPU (MPS)",
            "total_mem": "Shared with System"
        })

    # Storage
    total, used, free = shutil.disk_usage("/")
    info["storage_total"] = get_size(total)
    info["storage_free"] = get_size(free)

    return info

def get_recommended_profile(hw_info):
    """
    Returns recommended hyperparameters and a profile name based on hardware constraints.
    """
    profile = {
        "name": "Balanced Training",
        "device": "cpu",
        "batch_size": 64,
        "buffer_size": 100000,
        "n_envs": 1,
        "learning_rate": 0.0003,
        "notes": "Standard CPU training."
    }

    ram_total_gb = psutil.virtual_memory().total / (1024**3)

    if hw_info["cuda_available"] and len(hw_info["gpus"]) > 0:
        profile["name"] = "Fast Training (CUDA)"
        profile["device"] = "cuda"
        profile["batch_size"] = 256
        profile["buffer_size"] = 1000000 if ram_total_gb > 16 else 300000
        profile["n_envs"] = 4
        profile["notes"] = "CUDA enabled. Maximize batch size for GPU acceleration."
    elif hw_info["mps_available"]:
        profile["name"] = "macOS Optimized (MPS)"
        profile["device"] = "mps"
        profile["batch_size"] = 128
        profile["buffer_size"] = 500000 if ram_total_gb > 16 else 200000
        profile["n_envs"] = 1
        profile["notes"] = "Using Apple Silicon MPS. Keeping memory usage bounded."
    else:
        if hw_info["cpu_threads"] > 8 and ram_total_gb > 16:
            profile["name"] = "High-Performance CPU"
            profile["batch_size"] = 128
            profile["buffer_size"] = 500000
            profile["n_envs"] = hw_info["cpu_cores"] // 2
        elif psutil.sensors_battery() and not psutil.sensors_battery().power_plugged:
            profile["name"] = "Battery Saver"
            profile["batch_size"] = 32
            profile["buffer_size"] = 50000
            profile["learning_rate"] = 0.001 # faster convergence target, maybe less accurate
            profile["notes"] = "Running on battery. Reduced workload."
            
    return profile
