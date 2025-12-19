# ⚡️ Z-Image Distillation Trainer

<img width="1263" height="1021" alt="image" src="https://github.com/user-attachments/assets/79cc81bf-6084-4106-afe5-331f9be0750c" />



[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Single-file Python implementation of **Decoupled-DMD** (Distribution Matching Distillation) for creating few-step Z-Image models. Includes both a comprehensive CLI trainer and a full-featured Tkinter GUI.

<p align="center">
  <img src="https://img.shields.io/badge/RTX%205090-Optimized-76B900?logo=nvidia&logoColor=white" alt="RTX 5090 Optimized"/>
  <img src="https://img.shields.io/badge/Flash%20Attention-Supported-FF6B6B" alt="Flash Attention"/>
  <img src="https://img.shields.io/badge/LoRA-Supported-9B59B6" alt="LoRA Supported"/>
</p>

---

## 🌟 Features

### Core Capabilities
- ⚡ **Decoupled-DMD Algorithm**: CFG Augmentation + Distribution Matching for state-of-the-art distillation
- 🎯 **8-Step Inference**: Distill base models to run in 8 steps with comparable quality
- 🧠 **LoRA Support**: Memory-efficient training with PEFT
- 🚀 **RTX 5090 Optimized**: Flash Attention 2/3, BF16 precision, gradient checkpointing
- 🖥️ **Full GUI**: Tkinter-based interface with real-time monitoring
- 📊 **Real-time Metrics**: Live loss tracking, VRAM monitoring, validation previews

### Training Modes
- **Full Fine-Tuning**: Complete model retraining (48GB+ VRAM)
- **LoRA Training**: Parameter-efficient fine-tuning (16GB+ VRAM)
- **Quick Test**: Rapid iteration with 3 epochs
- **Production**: High-quality training with LPIPS loss

---

## 📦 Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/marduk191/z-image-distillation-trainer.git
cd z-image-distillation-trainer

# Install PyTorch with CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install diffusers from source (required for Z-Image support)
pip install git+https://github.com/huggingface/diffusers

# Install dependencies
pip install -r requirements.txt

# Optional: Flash Attention (highly recommended for RTX 5090)
pip install flash-attn --no-build-isolation
```

### Development Install

```bash
pip install -e ".[dev]"
```

---

## 🚀 Quick Start

### Option 1: GUI (Recommended for Beginners)

```bash
python launch_gui.py
```

Then:
1. Select a preset (Menu → Presets → "LoRA Training")
2. Browse for your training data
3. Click "▶ Start Training"
4. Monitor progress in real-time!

### Option 2: Command Line

```bash
python z_image_distillation_trainer.py \
  --teacher_model Tongyi-MAI/Z-Image-Base \
  --train_data_file examples/sample_training_data.json \
  --output_dir ./distillation_output \
  --use_lora \
  --lora_rank 64 \
  --num_epochs 10 \
  --train_batch_size 8 \
  --use_bf16 \
  --use_flash_attention
```

### Option 3: Interactive Script

```bash
chmod +x quick_start.sh
./quick_start.sh
```

---

## 📚 Documentation

- **[Package Overview](docs/PACKAGE_OVERVIEW.md)** - Quick reference guide
- **[Full README](README.md)** - Complete documentation with examples
- **[GUI Usage Guide](docs/GUI_USAGE_GUIDE.md)** - Comprehensive GUI tutorial
- **[GUI Features](docs/GUI_FEATURES.md)** - Technical feature overview

---

## 🎯 Training Data Format

Create a JSON file with your prompts:

```json
{
  "prompts": [
    {
      "prompt": "A serene mountain landscape at sunset",
      "negative_prompt": "blurry, low quality"
    },
    {
      "prompt": "Portrait of an elderly woman, natural lighting",
      "negative_prompt": "cartoon, anime"
    }
  ]
}
```

See [examples/sample_training_data.json](examples/sample_training_data.json) for more examples.

---

## 🎨 GUI Preview

The GUI provides:
- **Configuration Panel**: All training parameters in organized sections
- **Real-time Monitoring**: Live loss metrics, progress tracking
- **Validation Preview**: View generated images during training
- **System Monitoring**: GPU VRAM, CPU, RAM usage
- **Preset Configurations**: One-click optimal settings

---

## 🔧 Configuration Presets

### Full Fine-Tuning (48GB+ VRAM)
```bash
--train_batch_size 4 --learning_rate 1e-5 --num_epochs 10
```

### LoRA Training (16GB+ VRAM) ⭐ Recommended
```bash
--use_lora --lora_rank 64 --train_batch_size 8 --learning_rate 5e-5 --num_epochs 5
```

### Quick Test
```bash
--use_lora --lora_rank 32 --num_epochs 3 --max_train_prompts 100
```

### Production (High Quality)
```bash
--learning_rate 5e-6 --use_lpips --dm_weight 1.0 --num_epochs 20
```

---

## 📊 Expected Results

After training, you should achieve:
- **8-step inference** (vs 50-step base model)
- **Sub-second generation** on RTX 5090
- **Quality matching** original Z-Image-Turbo
- **VRAM usage**: <16GB with LoRA

---

## 🧪 Testing Your Model

```bash
# Test distilled model
python test_distilled_model.py --model_path ./distillation_output/final

# LoRA model
python test_distilled_model.py \
  --model_path ./distillation_output/final/lora \
  --is_lora

# Batch inference
python batch_inference.py \
  --model_path ./distillation_output/final \
  --prompt_file my_prompts.json \
  --output_dir ./test_results
```

---

## 💡 Key Features Explained

### Decoupled-DMD Algorithm

The training script implements the Decoupled Distribution Matching Distillation approach:

1. **CFG Augmentation (CA)** - Primary distillation engine
   - Augments teacher outputs with classifier-free guidance
   - Weight controlled by `--cfg_weight` (default: 1.0)

2. **Distribution Matching (DM)** - Regularization
   - Ensures output quality and stability
   - Weight controlled by `--dm_weight` (default: 0.5)
   - Optional LPIPS perceptual loss with `--use_lpips`

### Memory Optimization

Multiple strategies for VRAM efficiency:
- **LoRA Training**: 10x memory reduction
- **Gradient Checkpointing**: 2x memory reduction
- **BF16 Precision**: Better than FP16 for this architecture
- **Flash Attention**: Faster and more memory-efficient

---

## 🛠️ Troubleshooting

### CUDA Out of Memory
```bash
# Enable all memory optimizations
--gradient_checkpointing --use_lora --lora_rank 32 --train_batch_size 2
```

### Slow Training
```bash
# Enable performance optimizations
--use_flash_attention --use_bf16 --compile
```

### Poor Quality
```bash
# Increase regularization
--dm_weight 1.0 --use_lpips --learning_rate 5e-6
```

See [README.md](README.md) for detailed troubleshooting.

---

## 📁 Repository Structure

```
z-image-distillation-trainer/
├── z_image_distillation_trainer.py   # Main training script
├── z_image_distillation_gui.py       # GUI application
├── launch_gui.py                      # GUI launcher
├── test_distilled_model.py            # Model testing utility
├── batch_inference.py                 # Batch generation tool
├── quick_start.sh                     # Interactive setup script
├── requirements.txt                   # Dependencies
├── setup.py                           # Package installation
├── docs/                              # Documentation
│   ├── PACKAGE_OVERVIEW.md
│   ├── GUI_USAGE_GUIDE.md
│   └── GUI_FEATURES.md
└── examples/                          # Example configurations
    ├── sample_training_data.json
    └── config_example.yaml
```

---

## 📖 Citation

If you use this training script in your research or projects:

```bibtex
@article{liu2025decoupled,
  title={Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield},
  author={Dongyang Liu and Peng Gao and David Liu and Ruoyi Du and Zhen Li and Qilong Wu and Xin Jin and Sihan Cao and Shifeng Zhang and Hongsheng Li and Steven Hoi},
  journal={arXiv preprint arXiv:2511.22677},
  year={2025}
}

@article{team2025zimage,
  title={Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer},
  author={Z-Image Team},
  journal={arXiv preprint arXiv:2511.22699},
  year={2025}
}
```

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where we'd love help:
- Performance optimizations
- Additional loss functions
- Testing and validation
- Documentation improvements
- Bug reports and fixes

---

## 📄 License

This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) file for details.

Same as the Z-Image base model.

---

## 🙏 Acknowledgments

- **Tongyi-MAI team** for Z-Image model and research
- **DiffSynth-Studio** for training infrastructure insights
- **Anthropic** for development assistance
- **Community contributors** for feedback and improvements

---

## 🔗 Related Projects

- [Z-Image Official Repository](https://github.com/Tongyi-MAI/Z-Image)
- [Diffusers Library](https://github.com/huggingface/diffusers)
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=marduk191/z-image-distillation-trainer&type=Date)](https://star-history.com/#marduk191/z-image-distillation-trainer&Date)

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/marduk191/z-image-distillation-trainer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/marduk191/z-image-distillation-trainer/discussions)

---

<p align="center">
  Made with ❤️ for the Z-Image community
</p>

<p align="center">
  If this project helps you, please give it a ⭐!
</p>
