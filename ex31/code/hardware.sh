#!/usr/bin/env bash
# hardware.sh -- print the hardware and software facts relevant to training
# a tiny VLM, on either Linux or macOS.
# Usage: ./hardware.sh

set -u
OS="$(uname -s)"

hr() { printf '%s\n' "------------------------------------------------------------"; }

echo "tinyVLM hardware + environment check"
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
echo "  -- RAM mostly matters for image decoding + CLIP feature caching, not"
echo "     model size: the trainable parameters here fit in a few hundred MB."

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
echo "  -- device selection (cpu/cuda/mps) is a batch-size and wall-clock knob"
echo "     here, not a correctness one: config.demo.yaml's 350-step run was"
echo "     timed on CPU and does not require an accelerator."

# --- Disk --------------------------------------------------------------------
hr
echo "Disk (cwd)    :"
df -h . 2>/dev/null | tail -1 | awk '{print "   filesystem: " $1 ", used: " $3 ", avail: " $4 ", use%: " $5}'
echo "  -- budget: ~1.1 GB for Flickr8k images, ~600 MB for the CLIP ViT-B/32"
echo "     checkpoint, plus a few MB for training checkpoints and the holdout demo set."
for d in "./data" "./checkpoints" "./demo"; do
    if [ -e "$d" ]; then
        if [ -w "$d" ]; then
            echo "   $d: exists, writable"
        else
            echo "   $d: exists, NOT WRITABLE -- fix permissions before the talk"
        fi
    else
        echo "   $d: not created yet (trainVLM.py will create it)"
    fi
done

# --- Python / packages ----------------------------------------------------------
hr
if command -v python3 >/dev/null 2>&1; then
    echo "Python        : $(python3 --version 2>&1)"
    python3 - <<'PYEOF' 2>/dev/null
import importlib
required = ["torch", "transformers", "datasets", "sentencepiece", "yaml", "PIL", "matplotlib"]
missing = []
for mod in required:
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f"Packages      : MISSING -> {', '.join(missing)}")
    print("                install with: pip install torch torchvision transformers "
          "datasets sentencepiece pyyaml pillow matplotlib")
else:
    print("Packages      : torch, transformers, datasets, sentencepiece, pyyaml, pillow, "
          "matplotlib -- all present")

try:
    import torch
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    print(f"MPS available : {mps_ok}")
    if torch.cuda.is_available():
        print(f"CUDA device   : {torch.cuda.get_device_name(0)}")
except ImportError:
    pass
PYEOF
else
    echo "Python        : not found on PATH"
fi

# --- cache presence check --------------------------------------------------
hr
echo "Pre-downloaded assets:"
if [ -d "./data/flickr8k" ] && [ -n "$(ls -A ./data/flickr8k 2>/dev/null)" ]; then
    echo "   Flickr8k cache found under ./data/flickr8k"
else
    echo "   Flickr8k NOT found under ./data/flickr8k -- first trainVLM.py run will download it (~1.1 GB)"
fi
CLIP_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
# NOTE: -maxdepth must come right after the path and before any other test
# on BSD/macOS find (unlike GNU find, which tolerates it anywhere) -- this
# ordering works on both.
if [ -d "$CLIP_CACHE" ] && find "$CLIP_CACHE" -maxdepth 4 -iname "*clip-vit-base-patch32*" 2>/dev/null | grep -q .; then
    echo "   CLIP ViT-B/32 weights found in the Hugging Face cache"
    CLIP_CACHED=1
else
    echo "   CLIP ViT-B/32 weights NOT found in the Hugging Face cache -- first run will download them (~600 MB)"
    CLIP_CACHED=0
fi

# --- quick CLIP forward-pass benchmark --------------------------------------
hr
echo "CLIP forward-pass micro-benchmark, on the SAME device trainVLM.py would pick"
echo "(cuda > mps > cpu):"
if [ "$CLIP_CACHED" = "1" ]; then
    export TINYVLM_CLIP_CACHED=1
    python3 - <<'PYEOF' 2>/dev/null
import time
try:
    import torch
    # local_files_only=True: this benchmark must never trigger a network
    # download by itself -- if the weights aren't cached, fail fast and
    # say so, rather than silently fetching ~600 MB mid hardware-check.
    from transformers import CLIPModel, CLIPImageProcessor
    from PIL import Image
    import numpy as np

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name, local_files_only=True).to(device)
    proc = CLIPImageProcessor.from_pretrained(name, local_files_only=True)
    model.eval()
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    pixel_values = proc(images=img, return_tensors="pt")["pixel_values"].to(device)

    with torch.no_grad():
        model.get_image_features(pixel_values=pixel_values)  # warm-up
        n = 8
        t0 = time.time()
        for _ in range(n):
            model.get_image_features(pixel_values=pixel_values)
        dt = (time.time() - t0) / n
    print(f"   device: {device} | {dt*1000:.0f} ms/image (batch size 1)")
    print(f"   very roughly: {dt*16:.1f}s for one batch-of-16 CLIP forward pass before caching kicks in")
except Exception as e:
    print(f"   skipped ({type(e).__name__}: {e})")
PYEOF
else
    echo "   skipped -- CLIP weights not found in the local cache (see above); this"
    echo "   benchmark intentionally does not download them itself. Run trainVLM.py"
    echo "   once (or the prefetch command in the README) first, then re-run this script."
fi

hr
