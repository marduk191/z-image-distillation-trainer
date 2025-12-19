# Z-Image Distillation Training Package

Complete single-file Python implementation of Decoupled-DMD distillation for Z-Image models.

## 📦 Package Contents

### Core Files

1. **z_image_distillation_trainer.py** (Main Script)
   - Complete distillation training implementation
   - Decoupled-DMD algorithm (CFG Augmentation + Distribution Matching)
   - RTX 5090 optimized with Flash Attention support
   - LoRA and full fine-tuning modes
   - ~700 lines, fully documented

2. **requirements.txt**
   - All dependencies with version specifications
   - Includes optional packages (lpips, flash-attn)

3. **README.md**
   - Comprehensive documentation
   - Usage examples and troubleshooting
   - Configuration guide
   - Memory requirements

### Supporting Files

4. **quick_start.sh** (Interactive Setup)
   - Automated environment checking
   - Training mode selection wizard
   - One-command training launch

5. **test_distilled_model.py** (Quality Testing)
   - Test script for distilled models
   - Sample image generation
   - LoRA and full model support

6. **batch_inference.py** (Batch Processing)
   - Efficient batch image generation
   - Performance benchmarking
   - Metadata export

7. **config_example.yaml** (Configuration Template)
   - YAML configuration examples
   - Training presets
   - Advanced tuning tips

8. **sample_training_data.json** (Example Data)
   - 20 sample prompts
   - Proper JSON format
   - Diverse categories

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
# Install PyTorch with CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install diffusers from source (required for Z-Image)
pip install git+https://github.com/huggingface/diffusers

# Install other dependencies
pip install -r requirements.txt

# Optional: Flash Attention (highly recommended)
pip install flash-attn --no-build-isolation
```

### Step 2: Prepare Training Data
```bash
# Use the provided sample data
cp sample_training_data.json my_training_data.json

# Or create your own (see README for format)
```

### Step 3: Start Training
```bash
# Option A: Interactive wizard
./quick_start.sh

# Option B: Direct command
python z_image_distillation_trainer.py \
  --teacher_model Tongyi-MAI/Z-Image-Base \
  --train_data_file my_training_data.json \
  --output_dir ./output \
  --num_epochs 10 \
  --train_batch_size 4 \
  --use_lora \
  --use_bf16 \
  --use_flash_attention
```

## 📊 Training Modes

### Full Fine-Tuning (48GB+ VRAM)
- Complete model retraining
- Best quality, most flexible
- Memory intensive
```bash
python z_image_distillation_trainer.py \
  --train_data_file data.json \
  --train_batch_size 4 \
  --learning_rate 1e-5
```

### LoRA Training (16GB+ VRAM)
- Parameter-efficient fine-tuning
- 10x memory reduction
- Good quality, fast training
```bash
python z_image_distillation_trainer.py \
  --train_data_file data.json \
  --use_lora \
  --lora_rank 64 \
  --train_batch_size 8 \
  --learning_rate 5e-5
```

## 🧪 Testing & Validation

### Quick Test
```bash
python test_distilled_model.py \
  --model_path ./output/final \
  --output_dir ./test_results
```

### LoRA Test
```bash
python test_distilled_model.py \
  --model_path ./output/final/lora \
  --is_lora \
  --base_model Tongyi-MAI/Z-Image-Turbo
```

### Batch Inference
```bash
python batch_inference.py \
  --model_path ./output/final \
  --prompt_file test_prompts.json \
  --output_dir ./batch_results \
  --batch_size 4
```

## 📁 File Structure After Training

```
output/
├── checkpoint-1000/          # Periodic checkpoints
│   ├── lora/ or student/     # Model weights
│   ├── training_state.pt     # Optimizer state
│   └── args.json             # Training config
├── checkpoint-2000/
├── ...
├── final/                    # Final model
│   ├── lora/ or student/
│   └── training_state.pt
└── validation/               # Validation images
    ├── step-500/
    ├── step-1000/
    └── ...
```

## 🔧 Key Features

### Decoupled-DMD Algorithm
- **CFG Augmentation**: Primary distillation engine
- **Distribution Matching**: Quality regularization
- **Configurable weights**: Balance speed vs quality

### RTX 5090 Optimizations
- Flash Attention 2/3 support
- BF16 mixed precision
- Gradient checkpointing
- Efficient memory usage

### Production Ready
- Resume from checkpoints
- Validation during training
- Comprehensive error handling
- Extensive logging

## 💡 Common Use Cases

### 1. Create 8-Step Turbo Model
```bash
# Distill base model to 8 steps
python z_image_distillation_trainer.py \
  --teacher_model Tongyi-MAI/Z-Image-Base \
  --student_inference_steps 8 \
  --guidance_scale 7.5
```

### 2. Fine-Tune for Specific Style
```bash
# Use LoRA for style-specific fine-tuning
python z_image_distillation_trainer.py \
  --teacher_model Tongyi-MAI/Z-Image-Base \
  --train_data_file anime_prompts.json \
  --use_lora \
  --lora_rank 128
```

### 3. Optimize for Speed
```bash
# Maximize training speed
python z_image_distillation_trainer.py \
  --use_flash_attention \
  --use_bf16 \
  --compile \
  --train_batch_size 8
```

## ⚙️ Advanced Configuration

### Loss Tuning
```bash
# Emphasize quality over speed
--cfg_weight 1.0 --dm_weight 1.0 --use_lpips

# Emphasize speed over quality
--cfg_weight 2.0 --dm_weight 0.3

# Balanced approach (default)
--cfg_weight 1.0 --dm_weight 0.5
```

### Learning Rate Strategies
```bash
# Conservative (high quality)
--learning_rate 5e-6 --lr_warmup_steps 1000

# Aggressive (faster convergence)
--learning_rate 1e-4 --lr_warmup_steps 200

# Standard (balanced)
--learning_rate 1e-5 --lr_warmup_steps 500
```

## 🐛 Troubleshooting

### Out of Memory
```bash
# Enable gradient checkpointing
--gradient_checkpointing

# Use LoRA
--use_lora --lora_rank 32

# Reduce batch size
--train_batch_size 2

# Lower resolution
--resolution 768
```

### Poor Quality Results
```bash
# Increase DM weight
--dm_weight 1.0

# Enable LPIPS
--use_lpips

# Lower learning rate
--learning_rate 5e-6

# Train longer
--num_epochs 20
```

### Slow Training
```bash
# Enable Flash Attention
--use_flash_attention

# Use BF16
--use_bf16

# Reduce validation
--validation_steps 2000

# Increase batch size
--train_batch_size 8
```

## 📚 Resources

### Documentation
- **README.md**: Full documentation
- **config_example.yaml**: Configuration examples
- **Comments in code**: Inline explanations

### Papers
- Decoupled-DMD: arXiv:2511.22677
- Z-Image: arXiv:2511.22699
- DMDR: arXiv:2511.13649

### Community
- Z-Image GitHub: https://github.com/Tongyi-MAI/Z-Image
- Diffusers: https://github.com/huggingface/diffusers

## ✅ Validation Checklist

Before deploying your distilled model:

- [ ] Test with diverse prompts (people, objects, scenes)
- [ ] Check text rendering quality (both English and Chinese)
- [ ] Verify photorealism and detail preservation
- [ ] Test with challenging prompts (small text, complex scenes)
- [ ] Compare with original Turbo model
- [ ] Measure inference speed on target hardware
- [ ] Validate VRAM usage is within limits
- [ ] Check for artifacts or quality degradation

## 🎯 Next Steps

1. **Review README.md** for detailed documentation
2. **Run quick_start.sh** for guided setup
3. **Start with LoRA training** for quick iteration
4. **Test thoroughly** with test_distilled_model.py
5. **Scale to full training** once satisfied with LoRA results
6. **Integrate with ComfyUI** or your workflow

## 📝 Notes

- **Z-Image-Base** checkpoint not yet released (monitor official repo)
- **Flash Attention** highly recommended for RTX 5090
- **BF16 precision** preferred over FP16 for this architecture
- **LoRA training** recommended for first experiments
- **Validation images** generated during training for quality monitoring

## 🤝 Contributing

This is a standalone training script. For issues or improvements:
1. Test your changes thoroughly
2. Document any modifications
3. Share results with the community
4. Contribute back to Z-Image project

## 📄 License

Apache 2.0 - Same as Z-Image base model

---

**Ready to start?** Run `./quick_start.sh` or see README.md for detailed instructions!
