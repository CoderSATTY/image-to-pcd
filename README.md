# 🎯 Depth Anything V2 - Image to Point Cloud

Generate metric point clouds from single images using **Depth Anything V2**.

---

## 🚀 Quick Setup (One Command)

```bash
git clone https://github.com/CoderSATTY/image-to-pcd.git
cd image-to-pcd
chmod +x setup.sh && ./setup.sh
```

This downloads the model weights (~1.3GB) and sets up all dependencies.

---

## 📦 Usage

```bash
python depth_to_pointcloud.py \
  --encoder vitl \
  --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
  --max-depth 20 \
  --img-path ./input \
  --outdir ./output
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--encoder` | Model size: `vits`, `vitb`, `vitl` | `vitl` |
| `--load-from` | Path to model checkpoint | Required |
| `--max-depth` | Max depth in meters (20=indoor, 80=outdoor) | `20` |
| `--img-path` | Input image(s) path | Required |
| `--outdir` | Output directory for .ply files | Required |
| `--calibration` | Camera calibration file (.npz) | Optional |

---

## 📁 Project Structure

```
image-to-pcd/
├── setup.sh                    # Run this first!
├── depth_to_pointcloud.py      # Main script
├── checkpoints/                # Model weights (auto-downloaded)
│   └── depth_anything_v2_metric_hypersim_vitl.pth
├── depth_anything_v2/          # Model code (auto-cloned)
├── input/                      # Place your images here
└── output/                     # Point clouds saved here
```

---

## 🔗 Model Download Links (Manual)

If you prefer manual download:

| Model | Size | Download |
|-------|------|----------|
| Large (recommended) | 335M | [depth_anything_v2_metric_hypersim_vitl.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true) |
| Base | 97.5M | [depth_anything_v2_metric_hypersim_vitb.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth?download=true) |
| Small | 24.8M | [depth_anything_v2_metric_hypersim_vits.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true) |

Place in `checkpoints/` directory.

---

## 🙏 Acknowledgements

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) - NeurIPS 2024
