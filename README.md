# CUSTOM_CNN_FPGA

End-to-end toolkit for training CNN classifiers on cotton leaf diseases, exporting them to FPGA-friendly formats, and serving interactive demos. The project bundles TensorFlow/Keras, PyTorch, ONNX tooling, and hls4ml pipelines together with Docker-based workflows.

---

## ✨ Key features

- **Hybrid training suite**: `advanced_train.py`, `new_train.py`, and `train.py` cover Keras, PyTorch, and multi-model comparison workflows with live metric logging.
- **Deployment helpers**: `app.py` (Gradio) and `app_onnx.py` expose trained models; `netron_viewer.py` provides one-click Netron hosting for inspecting weights.
- **FPGA export path**: `generate_ip.py`, `convert_keras_to_ip.py`, and `hls4ml_*` scripts wire models into Vivado projects, ready for Zybo Z7-10 targets.
- **Batch job launcher**: `script_launcher.py` scans the repo and lets you execute any Python entry point interactively.

---

## 📁 Repository layout

```
images/                 # Dataset (bind-mounted into containers)
models/                 # Pre-trained weights and checkpoints
results/                # Training artefacts, plots, summaries
hls4ml_pytorch_project/ # Vivado + hls4ml firmware
fpga/                   # Board-specific TCL scripts
```

---

## 🧰 Requirements

### Host prerequisites

- Docker 24+
- Docker Compose V2 (`docker compose` CLI)
- NVIDIA driver 535+ (Linux) if you plan to use GPUs
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host

> **Note**: CPU-only usage is still supported. The GPU stack simply unlocks CUDA-accelerated training for TensorFlow and PyTorch.

### Python dependencies

`requirements.txt` now excludes the core deep-learning frameworks; the Docker build
installs them explicitly to avoid CUDA dependency conflicts:

1. `tensorflow[and-cuda]==2.16.1`
2. `torch==2.4.0+cu121`, `torchvision==0.19.0+cu121`, `torchaudio==2.4.0+cu121`

When the Docker image is built, these commands run in that order so TensorFlow brings
in the CUDA 12.3 runtime pieces while PyTorch reuses them without forcing a version
downgrade. Follow the same sequence if you install locally.

---

## 🚀 Run with Docker (recommended)

```
# Build a local image using the included Dockerfile
docker compose build

# Start the container and keep it running in the background
docker compose up -d

# Tail logs or attach to the interactive script launcher
docker compose logs -f
docker compose exec custom-cnn-fpga python script_launcher.py
```

The container mounts your `images/`, `models/`, and `results/` directories so training outputs persist on the host.

---

## ⚡ Enabling NVIDIA CUDA

1. **Verify host GPU stack**
   ```
   nvidia-smi
   ```
   Confirm the driver version matches the required CUDA runtime (12.3 works out-of-the-box).

2. **Build with GPU libraries**
   The Dockerfile already targets `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`, installs the CUDA-enabled TensorFlow/PyTorch wheels, and exports the appropriate `LD_LIBRARY_PATH`.

3. **Run containers with GPU access**
   `docker-compose.yml` specifies the NVIDIA runtime via `device_requests`. Ensure the NVIDIA Container Toolkit is present; then:
   ```
   docker compose up -d
   docker compose exec custom-cnn-fpga python - <<'PY'
   import tensorflow as tf, torch
   print('TF GPUs:', tf.config.list_physical_devices('GPU'))
   print('Torch CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
   PY
   ```

Common pitfalls:
- Missing `libnvidia-*.so` libraries ⇒ reinstall the matching driver/toolkit on the host.
- TensorFlow warning about TensorRT ⇒ optional; install NVIDIA TensorRT packages only if you need TF-TRT acceleration.

---

## 🧑‍💻 Local development (optional)

Prefer running outside Docker? Create a Python 3.10+ virtual environment:

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow[and-cuda]==2.16.1
pip install --no-deps --index-url https://download.pytorch.org/whl/cu121 \
   torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121
pip install -r requirements.txt
```

You still need CUDA libraries on the host to leverage the GPU builds.

---

## 📜 Useful scripts

| Script | Purpose |
| ------ | ------- |
| `script_launcher.py` | Interactive selector that runs any Python script in the repo |
| `advanced_train.py` | Multi-model training pipeline with live charts and summary reports |
| `training_monitor.py` | Streams training metrics to PNG dashboards |
| `generate_ip.py` / `convert_keras_to_ip.py` | Convert models into FPGA implementable IP cores |
| `app.py` / `app_onnx.py` | Launch Gradio demos for TensorFlow and ONNX runtimes |
| `netron_viewer.py` | Upload & host models for quick Netron inspection |

---

## 📈 Outputs & artefacts

All training runs populate the `results/` directory with:

- `<Model>_training.png` – accuracy/loss curves
- `<Model>_confusion.png` – confusion matrix
- `<Model>_report.json` – precision/recall/F1 summary
- `model_performance_summary.json` – aggregated view across models
- `model_accuracy_overview.png` – bar chart comparison

Clear the folder if you need a fresh analysis cache.

---

## 🛠️ Troubleshooting

- **TensorFlow cannot dlopen GPU libraries**: The host driver or container toolkit is mismatched. Reinstall the NVIDIA driver (`nvidia-smi` should succeed) and ensure `/usr/lib/libnvidia-*.so` exist.
- **Torch reports CPU-only**: Verify the CUDA 12.1 wheels were installed (`pip show torch` inside the container) and that `torch.cuda.is_available()` returns `True`.
- **Docker build fails fetching dependencies**: Ensure the host has network access and that corporate proxies are configured via `HTTP_PROXY`/`HTTPS_PROXY` build args when required.

---

## 📄 License

This project inherits the licensing terms defined in `LICENSE`.
