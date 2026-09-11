#!/usr/bin/env bash
# hardware.sh -- print the hardware facts relevant to training a tiny VLM,
# on either Linux or macOS. Meant to be run live at the start of the talk,
# right after kicking off trainVLM.py in another terminal tab.
#
# Usage: ./hardware.sh

set -u
OS="$(uname -s)"

hr() { printf '%s\n' "------------------------------------------------------------"; }

echo "tinyVLM hardware check"
hr
echo "OS            : $(uname -srm)"

# --- CPU -----------------------------------------------------------------
if [ "$OS" = "Darwin" ]; then
    CPU_BRAND="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    ARCH="$(uname -m)"
    if [ -z "$CPU_BRAND" ] && [ "$ARCH" = "arm64" ]; then
        CPU_BRAND="Apple Silicon ($ARCH)"
    fi
    PHYS_CORES="$(sysctl -n hw.physicalcpu 2>/dev/null)"
    LOGICAL_CORES="$(sysctl -n hw.logicalcpu 2>/dev/null)"
    echo "CPU           : ${CPU_BRAND:-unknown}"
    echo "CPU cores     : ${PHYS_CORES:-?} physical / ${LOGICAL_CORES:-?} logical"
elif [ "$OS" = "Linux" ]; then
    CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //')"
    PHYS_CORES="$(grep -c '^processor' /proc/cpuinfo 2>/dev/null)"
    echo "CPU           : ${CPU_MODEL:-unknown}"
    echo "CPU cores     : ${PHYS_CORES:-?} logical"
else
    echo "CPU           : unsupported OS ($OS) -- skipping"
fi

# --- RAM -------------------------------------------------------------------
if [ "$OS" = "Darwin" ]; then
    MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null)"
    if [ -n "$MEM_BYTES" ]; then
        echo "RAM total     : $(( MEM_BYTES / 1024 / 1024 / 1024 )) GB"
    fi
elif [ "$OS" = "Linux" ]; then
    MEM_KB="$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')"
    if [ -n "$MEM_KB" ]; then
        echo "RAM total     : $(( MEM_KB / 1024 / 1024 )) GB"
    fi
fi

# --- GPU / accelerator -------------------------------------------------------
echo "GPU / accel   :"
if [ "$OS" = "Darwin" ]; then
    system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|VRAM|Metal" | sed 's/^/   /'
    ARCH="$(uname -m)"
    if [ "$ARCH" = "arm64" ]; then
        echo "   Apple Silicon detected -> PyTorch MPS backend should be available"
    fi
elif [ "$OS" = "Linux" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/   /'
    else
        echo "   no NVIDIA GPU detected (nvidia-smi not found) -- will run on CPU"
    fi
fi

# --- Disk --------------------------------------------------------------------
hr
echo "Disk (cwd)    :"
df -h . 2>/dev/null | tail -1 | awk '{print "   filesystem: " $1 ", used: " $3 ", avail: " $4 ", use%: " $5}'

# --- Python / PyTorch ----------------------------------------------------------
hr
if command -v python3 >/dev/null 2>&1; then
    echo "Python        : $(python3 --version 2>&1)"
    python3 - <<'PYEOF' 2>/dev/null
try:
    import torch
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    print(f"MPS available : {mps_ok}")
    if torch.cuda.is_available():
        print(f"CUDA device   : {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch       : not installed (pip install torch)")
PYEOF
else
    echo "Python        : not found on PATH"
fi

hr
echo "Recommendation for this lecture's tiny VLM:"
echo "  - CPU-only is fine: the decoder is ~15M params, batch size 8-16."
echo "  - Apple Silicon (MPS) or any CUDA GPU speeds up the projector-only run"
echo "    from minutes to seconds per 100 steps, but is NOT required."
echo "  - Budget ~2 GB disk for Flickr8k + ~60 MB for the stories15M weights."

