set -euo pipefail
IFS=$'\n\t'

die() { echo "[✘] Error: $*" >&2; exit 1; }

if ! command -v conda &>/dev/null;        then die "conda is not available"; fi
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]];     then die "no conda environment activated"; fi
if [[ "$CONDA_DEFAULT_ENV" == base ]];    then die "activate a non-base conda environment"; fi
echo "[+] Using conda env: $CONDA_DEFAULT_ENV"

check_gpu() {
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi -L | grep -q 'GPU' \
      && { echo "[+] NVIDIA GPU detected"; return; }
  fi
  # fallback: PCI scan
  lspci | grep -qi 'NVIDIA' \
    && { echo "[+] NVIDIA GPU detected (via lspci)"; return; }
  die "No NVIDIA GPU detected or drivers unavailable"
}
check_gpu


ensure_cuda_stack() {
  local versions=("12.9.1" "12.9.0")
  local found_cudatoolkit=false
  
  echo "[+] Checking for existing CUDA toolkit installation..."
  
  # Check if any of the required cudatoolkit versions are installed
  for v in "${versions[@]}"; do
    # Check for cudatoolkit package
    if conda list --name "$CONDA_DEFAULT_ENV" 2>/dev/null | grep -q "^cudatoolkit\s\+${v}"; then
      echo "[+] Found cudatoolkit ${v}"
      found_cudatoolkit=true
      break
    fi
    # Also check for cuda-toolkit package (alternative naming)
    if conda list --name "$CONDA_DEFAULT_ENV" 2>/dev/null | grep -q "^cuda-toolkit\s\+${v}"; then
      echo "[+] Found cuda-toolkit ${v}"
      found_cudatoolkit=true
      break
    fi
  done
  
  if $found_cudatoolkit; then
    echo "[+] Compatible CUDA toolkit already present - proceeding"
    return 0
  fi
  
  # Neither version found, install both cuda and cudatoolkit 12.9.1
  echo "[+] No compatible CUDA toolkit found. Installing cuda=12.9.1 and cuda-toolkit=12.9.1..."
  
  conda install -y -c nvidia/label/cuda-12.9.1 cuda==12.9.1 cuda-toolkit==12.9.1
  
  if [ $? -eq 0 ]; then
    echo "[+] CUDA stack installation completed successfully"
  else
    echo "[!] CUDA stack installation failed"
    return 1
  fi
}
ensure_cuda_stack

# 1) Point all CUDA paths to the Conda toolkit
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export PATH="$CONDA_PREFIX/bin:${PATH:-}"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"


# 3) Set the compiler
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$CXX

# 4) Set the FA3 environment variables
export NVTE_CUDA_ARCHS="90;100"
export FLASH_ATTENTION_DISABLE_SPLIT='TRUE'
export FLASH_ATTENTION_DISABLE_PAGEDKV='TRUE'
export FLASH_ATTENTION_DISABLE_APPENDKV='TRUE'
export FLASH_ATTENTION_DISABLE_SOFTCAP='TRUE'
export FLASH_ATTENTION_DISABLE_PACKGQA='TRUE'
export FLASH_ATTENTION_DISABLE_FP16='TRUE'
export FLASH_ATTENTION_DISABLE_CLUSTER='TRUE'
export FLASH_ATTENTION_DISABLE_SM80='TRUE'

# 5) Build TE
source .venv/bin/activate

"$CUDACXX" --version | sed 's/^/[nvcc] /' || true
echo "[+] CUDA toolkit wired from conda ($CUDA_HOME)"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
PY

CUDNN_INC=$(python - <<'PY'
import site,glob
sp = site.getsitepackages()[0]
matches = glob.glob(sp + "/nvidia/cudnn*/include")
print(matches[0] if matches else "")
PY
)
CUDNN_LIB=$(python - <<'PY'
import site,glob
sp = site.getsitepackages()[0]
matches = glob.glob(sp + "/nvidia/cudnn*/lib*")
print(matches[0] if matches else "")
PY
)

[[ -n "$CUDNN_INC" && -f "$CUDNN_INC/cudnn.h" ]] || die "[!] cuDNN headers not found in uv venv (expected cudnn.h)"
[[ -n "$CUDNN_LIB" ]] || die "[!] cuDNN libs not found in uv venv"


export CPATH="$CUDNN_INC:$CPATH"
export CPLUS_INCLUDE_PATH="$CUDNN_INC:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="$CUDNN_LIB:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"
echo "[+] cuDNN wired: inc=$CUDNN_INC  lib=$CUDNN_LIB"

# optional archs (Hopper/Blackwell)
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90;100}"

uv pip install --no-build-isolation transformer_engine[pytorch]

python - <<'PY'
import transformer_engine.pytorch as te, torch
print("Transformer Engine import OK; torch:", torch.__version__)
PY

echo "[+] All good ✅"