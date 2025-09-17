FROM --platform=linux/amd64 pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
LABEL authors="john emeka, chika ojiako, nwaokeoma chidebube"

### system libs
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

### Python deps
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

### nnUNet
COPY tools tools/
RUN cd tools/nnUNet && pip3 install -e .

### GradScaler shim (PyTorch 2.2 drops the alias)
RUN python - <<'PY'
import textwrap, site, pathlib
path = pathlib.Path(site.getsitepackages()[0]) / "sitecustomize.py"
path.write_text(textwrap.dedent("""
    import torch
    if not hasattr(torch, 'GradScaler'):
        from torch.cuda.amp import GradScaler as _GS
        torch.GradScaler = _GS
"""))
PY

### project files
COPY checkpoint checkpoint/
COPY main.py .

CMD ["python", "main.py", "-i", "/input", "-o", "/output"]
