#!/bin/bash
# Z-Image Distillation Quick Start Script
# This script helps you get started with distillation training

set -e  # Exit on error

echo "======================================"
echo "Z-Image Distillation Quick Start"
echo "======================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_status "Python version: $python_version"

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    cuda_version=$(python3 -c "import torch; print(torch.version.cuda)")
    gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    gpu_memory=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')")
    print_status "CUDA available: $cuda_version"
    print_status "GPU: $gpu_name"
    print_status "VRAM: $gpu_memory"
else
    print_error "CUDA not available! GPU training requires CUDA."
    exit 1
fi

# Check if diffusers is installed with Z-Image support
echo ""
echo "Checking diffusers installation..."
if python3 -c "from diffusers import ZImagePipeline" 2>/dev/null; then
    print_status "Diffusers with Z-Image support installed"
else
    print_warning "Z-Image support not found in diffusers"
    echo "Installing latest diffusers from source..."
    pip install --quiet git+https://github.com/huggingface/diffusers
    print_status "Diffusers installed from source"
fi

# Check Flash Attention
echo ""
echo "Checking Flash Attention..."
if python3 -c "import flash_attn" 2>/dev/null; then
    print_status "Flash Attention installed"
    USE_FLASH_ATTN="--use_flash_attention"
else
    print_warning "Flash Attention not installed (optional but recommended for RTX 5090)"
    echo "Install with: pip install flash-attn --no-build-isolation"
    USE_FLASH_ATTN=""
fi

# Create output directory
OUTPUT_DIR="./distillation_output"
mkdir -p "$OUTPUT_DIR"
print_status "Output directory: $OUTPUT_DIR"

# Training mode selection
echo ""
echo "======================================"
echo "Select Training Mode:"
echo "======================================"
echo "1) Full Fine-Tuning (Requires 48GB+ VRAM)"
echo "2) LoRA Training (Requires 16GB+ VRAM)"
echo "3) Custom Configuration"
echo ""
read -p "Enter choice [1-3]: " mode_choice

case $mode_choice in
    1)
        print_status "Selected: Full Fine-Tuning"
        TRAINING_MODE="full"
        BATCH_SIZE=4
        LEARNING_RATE=1e-5
        USE_LORA=""
        ;;
    2)
        print_status "Selected: LoRA Training"
        TRAINING_MODE="lora"
        BATCH_SIZE=8
        LEARNING_RATE=5e-5
        USE_LORA="--use_lora --lora_rank 64 --lora_alpha 64"
        ;;
    3)
        print_status "Selected: Custom Configuration"
        read -p "Batch size [4]: " BATCH_SIZE
        BATCH_SIZE=${BATCH_SIZE:-4}
        read -p "Learning rate [1e-5]: " LEARNING_RATE
        LEARNING_RATE=${LEARNING_RATE:-1e-5}
        read -p "Use LoRA? [y/N]: " use_lora_input
        if [[ $use_lora_input == "y" || $use_lora_input == "Y" ]]; then
            USE_LORA="--use_lora --lora_rank 64 --lora_alpha 64"
        else
            USE_LORA=""
        fi
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

# Training data selection
echo ""
echo "======================================"
echo "Training Data:"
echo "======================================"
if [ -f "sample_training_data.json" ]; then
    print_status "Found sample_training_data.json"
    DEFAULT_DATA="sample_training_data.json"
else
    DEFAULT_DATA=""
fi

read -p "Training data file [$DEFAULT_DATA]: " TRAIN_DATA
TRAIN_DATA=${TRAIN_DATA:-$DEFAULT_DATA}

if [ ! -f "$TRAIN_DATA" ]; then
    print_error "Training data file not found: $TRAIN_DATA"
    exit 1
fi

# Number of epochs
read -p "Number of epochs [10]: " NUM_EPOCHS
NUM_EPOCHS=${NUM_EPOCHS:-10}

# Resolution
read -p "Training resolution [1024]: " RESOLUTION
RESOLUTION=${RESOLUTION:-1024}

# Build training command
echo ""
echo "======================================"
echo "Training Configuration:"
echo "======================================"
echo "Mode: $TRAINING_MODE"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "Data File: $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"
echo "======================================"
echo ""

read -p "Start training? [Y/n]: " start_training
if [[ $start_training == "n" || $start_training == "N" ]]; then
    print_warning "Training cancelled"
    exit 0
fi

# Construct the training command
TRAIN_CMD="python3 z_image_distillation_trainer.py \
  --teacher_model Tongyi-MAI/Z-Image-Base \
  --train_data_file $TRAIN_DATA \
  --output_dir $OUTPUT_DIR \
  --num_epochs $NUM_EPOCHS \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --resolution $RESOLUTION \
  --student_inference_steps 8 \
  --use_bf16 \
  --gradient_checkpointing \
  $USE_LORA \
  $USE_FLASH_ATTN"

echo ""
print_status "Starting training..."
echo ""
echo "Command: $TRAIN_CMD"
echo ""

# Run training
eval $TRAIN_CMD

# Training complete
echo ""
echo "======================================"
print_status "Training completed successfully!"
echo "======================================"
echo ""
echo "Model saved to: $OUTPUT_DIR/final"
echo ""
echo "To test your distilled model:"
echo "  python3 test_distilled_model.py --model_path $OUTPUT_DIR/final"
echo ""
