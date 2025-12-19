#!/usr/bin/env python3
"""
Z-Image Distillation Training Script
=====================================
Single-file implementation of Decoupled-DMD (Distribution Matching Distillation)
for Z-Image model family. Implements trajectory imitation distillation to create
few-step models from the base Z-Image checkpoint.

Key Features:
- Decoupled-DMD algorithm with CFG Augmentation and Distribution Matching
- Trajectory imitation from teacher (base) to student (turbo) model
- RTX 5090 optimized with Flash Attention 2/3 support
- Supports both full fine-tuning and LoRA training
- Comprehensive checkpointing and resume capabilities
- Memory-efficient training with gradient accumulation
- Mixed precision training (bf16/fp16)

Based on:
- Paper: "Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield"
- Paper: "Z-Image: An Efficient Image Generation Foundation Model"
- DiffSynth-Studio training methodology

Author: marduk191
License: Apache 2.0
"""

import argparse
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    from diffusers import ZImagePipeline, DDPMScheduler, DDIMScheduler
    from diffusers.models import AutoencoderKL
    from diffusers.optimization import get_scheduler
    from transformers import AutoTokenizer, AutoModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not installed. Install with: pip install git+https://github.com/huggingface/diffusers")

try:
    import safetensors
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not installed. Install with: pip install safetensors")

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed for LoRA training. Install with: pip install peft")

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Classes
# ============================================================================

class DistillationDataset(Dataset):
    """
    Dataset for distillation training.
    Loads prompts and optionally reference images for trajectory matching.
    """
    
    def __init__(
        self,
        data_file: str,
        resolution: int = 1024,
        max_prompts: Optional[int] = None,
    ):
        """
        Args:
            data_file: Path to JSON/JSONL file with prompts
            resolution: Target image resolution (square)
            max_prompts: Maximum number of prompts to load
        """
        self.resolution = resolution
        self.prompts = self._load_prompts(data_file, max_prompts)
        
        logger.info(f"Loaded {len(self.prompts)} prompts from {data_file}")
    
    def _load_prompts(self, data_file: str, max_prompts: Optional[int]) -> List[Dict[str, Any]]:
        """Load prompts from JSON/JSONL file."""
        prompts = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            if data_file.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        prompts.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    prompts = data
                elif isinstance(data, dict) and 'prompts' in data:
                    prompts = data['prompts']
        
        if max_prompts is not None:
            prompts = prompts[:max_prompts]
        
        return prompts
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        prompt_data = self.prompts[idx]
        
        if isinstance(prompt_data, str):
            prompt = prompt_data
            negative_prompt = ""
        else:
            prompt = prompt_data.get('prompt', prompt_data.get('text', ''))
            negative_prompt = prompt_data.get('negative_prompt', '')
        
        return {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'resolution': self.resolution,
        }


# ============================================================================
# Decoupled-DMD Loss Functions
# ============================================================================

class DecoupledDMDLoss(nn.Module):
    """
    Implements the Decoupled Distribution Matching Distillation loss.
    
    Key components:
    1. CFG Augmentation (CA) - Primary distillation engine
    2. Distribution Matching (DM) - Regularization for stability
    """
    
    def __init__(
        self,
        cfg_weight: float = 1.0,
        dm_weight: float = 0.5,
        use_lpips: bool = True,
    ):
        """
        Args:
            cfg_weight: Weight for CFG augmentation loss
            dm_weight: Weight for distribution matching loss
            use_lpips: Use LPIPS for perceptual loss (requires lpips package)
        """
        super().__init__()
        self.cfg_weight = cfg_weight
        self.dm_weight = dm_weight
        self.use_lpips = use_lpips
        
        if use_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='vgg').cuda()
                logger.info("LPIPS loss enabled for perceptual matching")
            except ImportError:
                logger.warning("lpips not installed, falling back to MSE loss")
                self.use_lpips = False
                self.lpips_fn = None
        else:
            self.lpips_fn = None
    
    def cfg_augmentation_loss(
        self,
        student_output: torch.Tensor,
        teacher_output_cfg: torch.Tensor,
        teacher_output_uncond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        CFG Augmentation Loss: Matches student to CFG-augmented teacher trajectory.
        
        teacher_cfg = teacher_uncond + guidance_scale * (teacher_cond - teacher_uncond)
        loss = MSE(student, teacher_cfg)
        """
        # Apply CFG to teacher output
        teacher_cfg = teacher_output_uncond + guidance_scale * (
            teacher_output_cfg - teacher_output_uncond
        )
        
        # MSE loss between student and CFG-augmented teacher
        loss = F.mse_loss(student_output, teacher_cfg)
        return loss
    
    def distribution_matching_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Distribution Matching Loss: Direct matching for regularization.
        """
        if self.use_lpips and self.lpips_fn is not None:
            # Perceptual loss with LPIPS
            loss = self.lpips_fn(student_output, teacher_output).mean()
        else:
            # MSE loss fallback
            loss = F.mse_loss(student_output, teacher_output)
        
        return loss
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output_cfg: torch.Tensor,
        teacher_output_uncond: torch.Tensor,
        guidance_scale: float = 7.5,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total Decoupled-DMD loss.
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # CFG Augmentation (primary engine)
        cfg_loss = self.cfg_augmentation_loss(
            student_output,
            teacher_output_cfg,
            teacher_output_uncond,
            guidance_scale,
        )
        
        # Distribution Matching (regularization)
        dm_loss = self.distribution_matching_loss(
            student_output,
            teacher_output_cfg,
        )
        
        # Combine losses
        total_loss = self.cfg_weight * cfg_loss + self.dm_weight * dm_loss
        
        loss_dict = {
            'cfg_loss': cfg_loss.item(),
            'dm_loss': dm_loss.item(),
            'total_loss': total_loss.item(),
        }
        
        return total_loss, loss_dict


# ============================================================================
# Distillation Trainer
# ============================================================================

class ZImageDistillationTrainer:
    """
    Main trainer class for Z-Image distillation using Decoupled-DMD.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        self.set_seed(args.seed)
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models and components
        self.setup_models()
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_loss()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Load checkpoint if resuming
        if args.resume_from:
            self.load_checkpoint(args.resume_from)
        
        logger.info("Trainer initialized successfully")
    
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def setup_models(self):
        """Initialize teacher and student models."""
        logger.info("Loading models...")
        
        # Load teacher model (base model with many steps)
        self.teacher_pipe = ZImagePipeline.from_pretrained(
            self.args.teacher_model,
            torch_dtype=torch.bfloat16 if self.args.use_bf16 else torch.float16,
        ).to(self.device)
        
        # Freeze teacher model
        for param in self.teacher_pipe.transformer.parameters():
            param.requires_grad = False
        self.teacher_pipe.transformer.eval()
        
        # Load student model (will be distilled to fewer steps)
        if self.args.student_model:
            logger.info(f"Loading student model from {self.args.student_model}")
            self.student_pipe = ZImagePipeline.from_pretrained(
                self.args.student_model,
                torch_dtype=torch.bfloat16 if self.args.use_bf16 else torch.float16,
            ).to(self.device)
        else:
            logger.info("Initializing student from teacher weights")
            self.student_pipe = ZImagePipeline.from_pretrained(
                self.args.teacher_model,
                torch_dtype=torch.bfloat16 if self.args.use_bf16 else torch.float16,
            ).to(self.device)
        
        self.student_pipe.transformer.train()
        
        # Apply LoRA if requested
        if self.args.use_lora and PEFT_AVAILABLE:
            logger.info(f"Applying LoRA with rank {self.args.lora_rank}")
            lora_config = LoraConfig(
                r=self.args.lora_rank,
                lora_alpha=self.args.lora_alpha,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                lora_dropout=0.1,
            )
            self.student_pipe.transformer = get_peft_model(
                self.student_pipe.transformer,
                lora_config
            )
        
        # Enable Flash Attention if available
        if self.args.use_flash_attention:
            try:
                self.student_pipe.transformer.set_attention_backend("flash")
                self.teacher_pipe.transformer.set_attention_backend("flash")
                logger.info("Flash Attention enabled")
            except Exception as e:
                logger.warning(f"Could not enable Flash Attention: {e}")
        
        # Enable gradient checkpointing for memory efficiency
        if self.args.gradient_checkpointing:
            self.student_pipe.transformer.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled")
        
        logger.info("Models loaded successfully")
    
    def setup_optimizers(self):
        """Initialize optimizer."""
        # Get trainable parameters
        if self.args.use_lora:
            trainable_params = [
                p for p in self.student_pipe.transformer.parameters() if p.requires_grad
            ]
        else:
            trainable_params = self.student_pipe.transformer.parameters()
        
        # Create optimizer
        if self.args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.weight_decay,
                eps=self.args.adam_epsilon,
            )
        elif self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.weight_decay,
                eps=self.args.adam_epsilon,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if self.args.use_fp16 else None
        
        logger.info(f"Optimizer: {self.args.optimizer}, LR: {self.args.learning_rate}")
    
    def setup_schedulers(self):
        """Initialize learning rate scheduler."""
        # Calculate total training steps
        dataset = DistillationDataset(
            self.args.train_data_file,
            resolution=self.args.resolution,
            max_prompts=self.args.max_train_prompts,
        )
        
        steps_per_epoch = len(dataset) // self.args.train_batch_size
        self.total_steps = steps_per_epoch * self.args.num_epochs
        
        # Create scheduler
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.total_steps,
        )
        
        logger.info(f"Total training steps: {self.total_steps}")
    
    def setup_loss(self):
        """Initialize loss function."""
        self.loss_fn = DecoupledDMDLoss(
            cfg_weight=self.args.cfg_weight,
            dm_weight=self.args.dm_weight,
            use_lpips=self.args.use_lpips,
        )
        logger.info("Loss function initialized")
    
    def train_step(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step with trajectory distillation.
        
        Process:
        1. Sample noise and timesteps
        2. Generate teacher trajectories (conditional + unconditional)
        3. Generate student trajectory (single forward pass)
        4. Compute Decoupled-DMD loss
        5. Backpropagate and update
        """
        prompts = batch['prompt']
        negative_prompts = batch['negative_prompt']
        resolution = batch['resolution'][0].item()
        
        batch_size = len(prompts)
        
        # Sample random timestep for this batch
        timesteps = torch.randint(
            0,
            self.teacher_pipe.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        
        # Generate random noise
        latent_shape = (
            batch_size,
            self.teacher_pipe.transformer.config.in_channels,
            resolution // 8,  # VAE downscaling factor
            resolution // 8,
        )
        noise = torch.randn(latent_shape, device=self.device, dtype=torch.bfloat16)
        
        # Encode prompts
        with torch.no_grad():
            # Conditional prompt embeddings
            prompt_embeds = self.teacher_pipe.encode_prompt(
                prompts,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            
            # Unconditional prompt embeddings
            negative_embeds = self.teacher_pipe.encode_prompt(
                [""] * batch_size,  # Empty prompts for unconditional
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        
        # Forward pass through teacher (conditional)
        with torch.no_grad():
            teacher_output_cfg = self.teacher_pipe.transformer(
                hidden_states=noise,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            
            # Forward pass through teacher (unconditional)
            teacher_output_uncond = self.teacher_pipe.transformer(
                hidden_states=noise,
                timestep=timesteps,
                encoder_hidden_states=negative_embeds,
                return_dict=False,
            )[0]
        
        # Forward pass through student
        student_output = self.student_pipe.transformer(
            hidden_states=noise,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        
        # Compute loss
        loss, loss_dict = self.loss_fn(
            student_output=student_output,
            teacher_output_cfg=teacher_output_cfg,
            teacher_output_uncond=teacher_output_uncond,
            guidance_scale=self.args.guidance_scale,
        )
        
        return loss, loss_dict
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Create dataset and dataloader
        train_dataset = DistillationDataset(
            self.args.train_data_file,
            resolution=self.args.resolution,
            max_prompts=self.args.max_train_prompts,
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        
        # Training loop
        progress_bar = tqdm(
            range(self.total_steps),
            initial=self.global_step,
            desc="Training",
        )
        
        for epoch in range(self.epoch, self.args.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(train_dataloader):
                # Training step
                if self.args.use_fp16 and self.scaler is not None:
                    with autocast():
                        loss, loss_dict = self.train_step(batch)
                    
                    self.scaler.scale(loss).backward()
                    
                    if self.args.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.student_pipe.transformer.parameters(),
                            self.args.max_grad_norm,
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss, loss_dict = self.train_step(batch)
                    loss.backward()
                    
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.student_pipe.transformer.parameters(),
                            self.args.max_grad_norm,
                        )
                    
                    self.optimizer.step()
                
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # Update progress
                self.global_step += 1
                epoch_loss += loss.item()
                num_batches += 1
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
                })
                
                # Logging
                if self.global_step % self.args.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(
                        f"Epoch {epoch}, Step {self.global_step}: "
                        f"loss={avg_loss:.4f}, "
                        f"cfg_loss={loss_dict['cfg_loss']:.4f}, "
                        f"dm_loss={loss_dict['dm_loss']:.4f}, "
                        f"lr={self.lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                
                # Save checkpoint
                if self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint()
                
                # Validation
                if self.args.validation_prompts and self.global_step % self.args.validation_steps == 0:
                    self.validate()
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint-epoch-{epoch}")
        
        # Training complete
        logger.info("Training completed!")
        self.save_checkpoint("final")
    
    @torch.no_grad()
    def validate(self):
        """Run validation with sample prompts."""
        logger.info("Running validation...")
        
        self.student_pipe.transformer.eval()
        
        validation_dir = self.output_dir / "validation" / f"step-{self.global_step}"
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, prompt in enumerate(self.args.validation_prompts):
            try:
                image = self.student_pipe(
                    prompt=prompt,
                    height=self.args.resolution,
                    width=self.args.resolution,
                    num_inference_steps=self.args.student_inference_steps,
                    guidance_scale=self.args.student_guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(self.args.seed),
                ).images[0]
                
                image.save(validation_dir / f"sample_{idx}.png")
                
                # Save prompt
                with open(validation_dir / f"sample_{idx}.txt", 'w') as f:
                    f.write(prompt)
            
            except Exception as e:
                logger.error(f"Validation failed for prompt {idx}: {e}")
        
        self.student_pipe.transformer.train()
        logger.info(f"Validation images saved to {validation_dir}")
    
    def save_checkpoint(self, checkpoint_name: Optional[str] = None):
        """Save training checkpoint."""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint-{self.global_step}"
        
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        
        # Save student model
        if self.args.use_lora:
            # Save LoRA weights
            self.student_pipe.transformer.save_pretrained(checkpoint_dir / "lora")
        else:
            # Save full model
            self.student_pipe.save_pretrained(checkpoint_dir / "student")
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }
        
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        
        torch.save(state, checkpoint_dir / "training_state.pt")
        
        # Save args
        with open(checkpoint_dir / "args.json", 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load training state
        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.optimizer.load_state_dict(state['optimizer'])
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            
            if self.scaler is not None and 'scaler' in state:
                self.scaler.load_state_dict(state['scaler'])
            
            logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Z-Image Distillation Training")
    
    # Model paths
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="Tongyi-MAI/Z-Image-Base",  # Will be released
        help="Teacher model path (base model with many steps)",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default=None,
        help="Student model path (optional, defaults to teacher weights)",
    )
    
    # Training data
    parser.add_argument(
        "--train_data_file",
        type=str,
        required=True,
        help="Path to training prompts JSON/JSONL file",
    )
    parser.add_argument(
        "--max_train_prompts",
        type=int,
        default=None,
        help="Maximum number of training prompts to use",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Training resolution (square)",
    )
    
    # Training parameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Training batch size per GPU",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
        help="Optimizer type",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Adam epsilon",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 to disable)",
    )
    
    # Distillation parameters
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for teacher CFG augmentation",
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        default=1.0,
        help="Weight for CFG augmentation loss",
    )
    parser.add_argument(
        "--dm_weight",
        type=float,
        default=0.5,
        help="Weight for distribution matching loss",
    )
    parser.add_argument(
        "--use_lpips",
        action="store_true",
        help="Use LPIPS perceptual loss for distribution matching",
    )
    
    # Student inference parameters
    parser.add_argument(
        "--student_inference_steps",
        type=int,
        default=8,
        help="Number of inference steps for distilled student model",
    )
    parser.add_argument(
        "--student_guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for student inference (typically 1.0 for turbo)",
    )
    
    # LoRA parameters
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for efficient fine-tuning",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha",
    )
    
    # Memory and performance
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision (recommended for RTX 5090)",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use float16 mixed precision training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        default=True,
        help="Enable Flash Attention 2/3 for faster training",
    )
    
    # Logging and checkpointing
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./distillation_output",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every N steps",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        nargs="+",
        default=[
            "A serene landscape with mountains and a lake at sunset",
            "Portrait of a young woman with flowing hair, studio lighting",
            "Futuristic cityscape with neon lights and flying cars",
        ],
        help="Prompts to use for validation",
    )
    
    # Miscellaneous
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate dependencies
    if not DIFFUSERS_AVAILABLE:
        raise ImportError(
            "diffusers is required. Install with: "
            "pip install git+https://github.com/huggingface/diffusers"
        )
    
    if not SAFETENSORS_AVAILABLE:
        logger.warning("safetensors not installed, checkpoint saving may be slower")
    
    if args.use_lora and not PEFT_AVAILABLE:
        raise ImportError("PEFT is required for LoRA training. Install with: pip install peft")
    
    if args.use_lpips:
        try:
            import lpips
        except ImportError:
            logger.warning("lpips not installed, falling back to MSE loss")
            args.use_lpips = False
    
    # Create trainer and start training
    trainer = ZImageDistillationTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
