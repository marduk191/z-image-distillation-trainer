# Z-Image Distillation Trainer - GUI User Guide

Complete guide for using the Tkinter GUI interface for Z-Image distillation training.

## 🖥️ GUI Overview

The GUI provides an intuitive interface for:
- **Easy parameter configuration** with visual controls
- **Real-time training monitoring** with live metrics
- **VRAM and system resource tracking**
- **Validation image preview** during training
- **Configuration presets** for common use cases
- **Save/load configurations** for reproducibility

## 🚀 Quick Start

### 1. Launch the GUI

```bash
# Option 1: Direct launch
python z_image_distillation_gui.py

# Option 2: Using launcher (checks dependencies)
python launch_gui.py
```

### 2. Configure Training

1. **Select a Preset** (Menu → Presets):
   - Full Fine-Tuning (48GB VRAM)
   - LoRA Training (16GB VRAM) ← **Recommended for first try**
   - Quick Test (3 Epochs)
   - Production (High Quality)

2. **Adjust Parameters** in the left panel:
   - Model paths
   - Training data
   - Hyperparameters
   - Performance settings

3. **Click "▶ Start Training"**

### 3. Monitor Progress

Switch between tabs in the right panel:
- **Progress**: Training metrics and time estimates
- **Logs**: Real-time training output
- **Validation**: Preview generated images
- **System**: GPU/CPU/RAM monitoring

## 📋 GUI Layout

### Left Panel - Configuration

#### 1. Model Configuration
- **Teacher Model**: Base model path (e.g., `Tongyi-MAI/Z-Image-Base`)
- **Student Model**: Optional custom student initialization

#### 2. Data Configuration
- **Training Data**: JSON/JSONL file with prompts (Browse button available)
- **Max Prompts**: Limit training data size for quick tests
- **Resolution**: Image resolution (512, 768, 1024)

#### 3. Training Parameters
- **Epochs**: Number of training epochs
- **Batch Size**: Samples per batch (adjust based on VRAM)
- **Learning Rate**: Training learning rate
- **LR Scheduler**: Learning rate schedule type
- **Warmup Steps**: Warmup period for learning rate

#### 4. Distillation Settings
- **Teacher CFG Scale**: Guidance scale for teacher model (typically 7.5)
- **CFG Weight**: Weight for CFG augmentation loss (primary engine)
- **DM Weight**: Weight for distribution matching loss (regularizer)
- **Target Steps**: Inference steps for distilled model (typically 8)
- **Use LPIPS**: Enable perceptual loss for better quality

#### 5. LoRA Settings
- **Enable LoRA**: Toggle LoRA training (memory efficient)
- **LoRA Rank**: LoRA matrix rank (higher = more capacity)
- **LoRA Alpha**: LoRA scaling factor (typically equals rank)

#### 6. Performance & Memory
- **Use BF16**: BF16 mixed precision (recommended for RTX 5090)
- **Use Flash Attention**: Flash Attention 2/3 for faster training
- **Gradient Checkpointing**: Save memory at cost of speed

#### 7. Output & Logging
- **Output Dir**: Directory for checkpoints and validation images
- **Log Every N Steps**: Logging frequency
- **Save Every N Steps**: Checkpoint frequency
- **Validate Every N Steps**: Validation image generation frequency

### Right Panel - Monitoring

#### Progress Tab
- **Current Epoch/Step**: Training progress
- **Progress Bar**: Visual progress indicator
- **Loss Metrics**: Real-time loss values
  - Total Loss
  - CFG Loss (primary engine)
  - DM Loss (regularizer)
  - Learning Rate
- **Time Estimate**: Elapsed and remaining time

#### Logs Tab
- **Real-time Output**: Streaming training logs
- **Clear Button**: Clear log history

#### Validation Tab
- **Image Preview**: View generated validation images
- **Navigation**: Previous/Next buttons
- **Refresh**: Update image list

#### System Tab
- **GPU Information**: GPU name and VRAM usage
- **System Information**: CPU and RAM usage

## 🎮 Using Presets

Presets provide optimized configurations for common scenarios:

### Full Fine-Tuning (48GB+ VRAM)
```
Batch Size: 4
Learning Rate: 1e-5
LoRA: Disabled
Epochs: 10
Best for: Maximum quality, full model retraining
```

### LoRA Training (16GB+ VRAM) ⭐ Recommended
```
Batch Size: 8
Learning Rate: 5e-5
LoRA: Enabled (Rank 64)
Epochs: 5
Best for: Memory-efficient, good quality
```

### Quick Test (3 Epochs)
```
Batch Size: 8
Learning Rate: 5e-5
LoRA: Enabled (Rank 32)
Epochs: 3
Max Prompts: 100
Best for: Quick iteration, testing configurations
```

### Production (High Quality)
```
Batch Size: 4
Learning Rate: 5e-6
LoRA: Disabled
LPIPS: Enabled
DM Weight: 1.0
Epochs: 20
Best for: Final production models
```

## 💾 Configuration Management

### Save Configuration
1. Set up your desired parameters
2. Menu → File → Save Configuration
3. Choose filename (e.g., `my_config.json`)

### Load Configuration
1. Menu → File → Load Configuration
2. Select saved JSON file
3. All parameters updated automatically

### New Configuration
1. Menu → File → New Configuration
2. Resets to default settings

## 📊 Understanding Metrics

### Loss Values

**Total Loss** (Combined loss)
- Good: 0.03 - 0.06
- Excellent: < 0.03
- Poor: > 0.10

**CFG Loss** (Primary distillation engine)
- Should decrease steadily
- Target: < 0.01 by end of training

**DM Loss** (Regularization)
- Stabilizes quality
- Target: 0.02 - 0.05

**Learning Rate**
- Decreases over training with scheduler
- Sudden changes indicate warmup/schedule steps

### Progress Indicators

**Progress Bar**: Shows completion percentage
**Epoch**: Current/Total epochs
**Step**: Current/Total steps
**Elapsed Time**: Time since training started
**Remaining Time**: Estimated time to completion (approximate)

## 🎨 Validation Images

During training, the GUI generates validation images at regular intervals:

1. **Automatic Generation**: Based on "Validate Every N Steps" setting
2. **View in GUI**: Switch to "Validation" tab
3. **Navigate**: Use Previous/Next buttons
4. **Location**: Saved to `output_dir/validation/step-XXXX/`

### What to Look For

✅ **Good Signs**:
- Clear, detailed images
- Good prompt adherence
- Proper color and composition
- No obvious artifacts

❌ **Warning Signs**:
- Blurry or distorted images
- Poor prompt following
- Color shifts or artifacts
- Quality degradation over time

## 🛠️ Training Controls

### Start Training
- **Button**: ▶ Start Training
- **Action**: Launches training with current configuration
- **Confirmation**: Shows confirmation dialog

### Stop Training
- **Button**: ⏹ Stop Training
- **Action**: Gracefully terminates training
- **Note**: Current epoch will be interrupted

### Test Model
- **Button**: 🧪 Test Model
- **Action**: Launches test script with trained model
- **Requirement**: Completed training checkpoint exists

## 🔧 Troubleshooting

### GUI Won't Start

**Problem**: Missing tkinter
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS (usually included)
# Windows (usually included)
```

**Problem**: Missing PIL
```bash
pip install Pillow
```

### Training Won't Start

**Error**: "Training data file does not exist"
- **Solution**: Click Browse button and select valid JSON/JSONL file

**Error**: CUDA out of memory
- **Solution**: 
  1. Enable LoRA training
  2. Reduce batch size
  3. Enable gradient checkpointing
  4. Lower resolution

### No Validation Images

**Problem**: Validation tab is empty
- **Solution**: 
  1. Click "🔄 Refresh" button
  2. Check output directory exists
  3. Wait for validation step (based on validation_steps setting)

### Logs Not Updating

**Problem**: Log display frozen
- **Solution**: Training might be at a non-logging step
- **Check**: Wait for next logging interval (logging_steps setting)

## 📁 Output Structure

Training creates this directory structure:

```
output_dir/
├── checkpoint-1000/
│   ├── lora/ (if LoRA) or student/ (if full)
│   │   └── *.safetensors
│   ├── training_state.pt
│   └── args.json
├── checkpoint-2000/
│   └── ...
├── final/
│   ├── lora/ or student/
│   └── training_state.pt
└── validation/
    ├── step-500/
    │   ├── sample_0.png
    │   ├── sample_0.txt
    │   └── ...
    ├── step-1000/
    └── ...
```

## 🎯 Best Practices

### First Time Users

1. **Start with LoRA preset** (Menu → Presets → LoRA Training)
2. **Use small dataset** (100-500 prompts) for testing
3. **Monitor first few steps** carefully in Logs tab
4. **Check validation images** early (step 500)
5. **Adjust if needed** and restart

### Memory Management

**If running out of VRAM**:
1. ☑️ Enable Gradient Checkpointing
2. ☑️ Enable LoRA Training
3. ⬇️ Reduce Batch Size (4 → 2 → 1)
4. ⬇️ Lower Resolution (1024 → 768)

**Optimal for RTX 5090**:
- ☑️ Use BF16 Precision
- ☑️ Use Flash Attention
- ☑️ Gradient Checkpointing
- Batch Size: 8 (LoRA) or 4 (Full)

### Quality Optimization

**For best quality**:
1. Use Production preset
2. Enable LPIPS perceptual loss
3. Increase DM weight to 1.0
4. Train for more epochs (20+)
5. Use diverse, high-quality training data

**For faster training**:
1. Use Quick Test preset
2. Disable LPIPS
3. Use LoRA with smaller rank (32)
4. Fewer epochs (3-5)

## 🔍 System Monitoring

### GPU Monitoring
- **VRAM Usage**: Current/Total memory usage
- **Progress Bar**: Visual VRAM percentage
- **Update**: Every 2 seconds during training

### System Monitoring
- **CPU Usage**: Current CPU load percentage
- **RAM Usage**: Current/Total system memory
- **Update**: Every 2 seconds

## 💡 Tips & Tricks

### Configuration Tips

1. **Save configurations** for reproducibility
2. **Use presets** as starting points
3. **Document changes** in config filenames
4. **Test small first** before full training

### Training Tips

1. **Monitor early validation** (step 500-1000)
2. **Watch for loss convergence** in Progress tab
3. **Check logs** for errors or warnings
4. **Save intermediate checkpoints** (every 1000 steps)

### Validation Tips

1. **Use diverse prompts** for validation
2. **Include challenging cases** (small text, complex scenes)
3. **Compare with base model** outputs
4. **Save good validation images** for reference

## 🎓 Advanced Usage

### Multi-Configuration Workflow

1. Save baseline config: `baseline.json`
2. Test variations:
   - `baseline_high_lr.json`
   - `baseline_low_cfg.json`
   - `baseline_lora128.json`
3. Compare results
4. Select best configuration

### Training Sessions

For long training runs:
1. Start with Quick Test preset
2. Validate approach works
3. Switch to Production preset
4. Run overnight
5. Check validation images in morning

### Checkpoint Management

- Keep important checkpoints
- Delete intermediate checkpoints to save space
- Use meaningful checkpoint names
- Document best checkpoint in notes

## 📚 Related Tools

After training in GUI:
- **Test with**: `test_distilled_model.py`
- **Batch inference**: `batch_inference.py`
- **Command line**: `z_image_distillation_trainer.py`

## 🆘 Getting Help

1. Check this guide first
2. Review main README.md
3. Check training logs for errors
4. Consult Z-Image GitHub issues
5. Review Decoupled-DMD paper for methodology

## 📝 Notes

- **Training time** varies based on hardware, dataset size, and parameters
- **VRAM requirements** depend on resolution, batch size, and LoRA usage
- **Quality improves** with more epochs and better training data
- **Validation images** help monitor training quality
- **Checkpoints allow** resuming interrupted training

---

**Enjoy distilling your Z-Image models!** 🎨✨

For technical details, see the main README.md and paper references.
