#!/usr/bin/env python3
"""
Z-Image Distillation Trainer GUI
=================================
Tkinter-based graphical interface for Z-Image distillation training.

Features:
- Easy parameter configuration with presets
- Real-time training progress monitoring
- VRAM usage tracking
- Validation image preview
- Training history and logs
- Save/load configurations
- One-click training launch

Author: marduk191
License: Apache 2.0
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import *
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Optional, Dict, Any

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not installed. Image preview disabled.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed. VRAM monitoring disabled.")


class ZImageDistillationGUI:
    """Main GUI application for Z-Image distillation training."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Z-Image Distillation Trainer")
        self.root.geometry("1400x900")
        
        # Training state
        self.training_process: Optional[subprocess.Popen] = None
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        
        # Configuration
        self.config = self.load_default_config()
        
        # Setup UI
        self.setup_styles()
        self.setup_menu()
        self.setup_ui()
        
        # Start monitoring
        self.start_monitoring()
        
    def setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom styles
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Section.TLabelframe', font=('Arial', 10, 'bold'))
        style.configure('Success.TButton', background='#28a745')
        style.configure('Danger.TButton', background='#dc3545')
        style.configure('Primary.TButton', background='#007bff')
    
    def setup_menu(self):
        """Setup menu bar."""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Configuration", command=self.new_config)
        file_menu.add_command(label="Load Configuration", command=self.load_config)
        file_menu.add_command(label="Save Configuration", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Presets menu
        presets_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Presets", menu=presets_menu)
        presets_menu.add_command(label="Full Fine-Tuning (48GB VRAM)", command=lambda: self.apply_preset("full"))
        presets_menu.add_command(label="LoRA Training (16GB VRAM)", command=lambda: self.apply_preset("lora"))
        presets_menu.add_command(label="Quick Test (3 Epochs)", command=lambda: self.apply_preset("test"))
        presets_menu.add_command(label="Production (High Quality)", command=lambda: self.apply_preset("production"))
        
        # Help menu
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_docs)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_ui(self):
        """Setup main UI layout."""
        # Main container with paned window
        main_paned = ttk.PanedWindow(self.root, orient=HORIZONTAL)
        main_paned.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Configuration
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=3)
        
        # Right panel - Monitoring
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        # Setup left panel
        self.setup_config_panel(left_frame)
        
        # Setup right panel
        self.setup_monitoring_panel(right_frame)
        
        # Status bar
        self.setup_status_bar()
    
    def setup_config_panel(self, parent):
        """Setup configuration panel with scrollable canvas."""
        # Canvas with scrollbar
        canvas = Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient=VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # Bind mouse wheel
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        # Configuration sections
        row = 0
        
        # Model Configuration
        row = self.setup_model_section(scrollable_frame, row)
        
        # Data Configuration
        row = self.setup_data_section(scrollable_frame, row)
        
        # Training Parameters
        row = self.setup_training_section(scrollable_frame, row)
        
        # Distillation Settings
        row = self.setup_distillation_section(scrollable_frame, row)
        
        # LoRA Settings
        row = self.setup_lora_section(scrollable_frame, row)
        
        # Performance Settings
        row = self.setup_performance_section(scrollable_frame, row)
        
        # Output Settings
        row = self.setup_output_section(scrollable_frame, row)
        
        # Control buttons
        self.setup_control_buttons(scrollable_frame, row)
    
    def setup_model_section(self, parent, row):
        """Setup model configuration section."""
        frame = ttk.LabelFrame(parent, text="Model Configuration", style='Section.TLabelframe')
        frame.grid(row=row, column=0, sticky=(W, E), padx=5, pady=5)
        
        # Teacher model
        ttk.Label(frame, text="Teacher Model:").grid(row=0, column=0, sticky=W, padx=5, pady=2)
        self.teacher_model_var = StringVar(value=self.config['teacher_model'])
        ttk.Entry(frame, textvariable=self.teacher_model_var, width=50).grid(row=0, column=1, padx=5, pady=2)
        
        # Student model
        ttk.Label(frame, text="Student Model:").grid(row=1, column=0, sticky=W, padx=5, pady=2)
        self.student_model_var = StringVar(value=self.config.get('student_model', ''))
        ttk.Entry(frame, textvariable=self.student_model_var, width=50).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(frame, text="(Optional - defaults to teacher)", font=('Arial', 8, 'italic')).grid(row=2, column=1, sticky=W, padx=5)
        
        return row + 1
    
    def setup_data_section(self, parent, row):
        """Setup data configuration section."""
        frame = ttk.LabelFrame(parent, text="Data Configuration", style='Section.TLabelframe')
        frame.grid(row=row, column=0, sticky=(W, E), padx=5, pady=5)
        
        # Training data file
        ttk.Label(frame, text="Training Data:").grid(row=0, column=0, sticky=W, padx=5, pady=2)
        data_frame = ttk.Frame(frame)
        data_frame.grid(row=0, column=1, sticky=(W, E), padx=5, pady=2)
        
        self.train_data_var = StringVar(value=self.config['train_data_file'])
        ttk.Entry(data_frame, textvariable=self.train_data_var, width=40).pack(side=LEFT, padx=2)
        ttk.Button(data_frame, text="Browse...", command=self.browse_train_data).pack(side=LEFT)
        
        # Max prompts
        ttk.Label(frame, text="Max Prompts:").grid(row=1, column=0, sticky=W, padx=5, pady=2)
        self.max_prompts_var = StringVar(value=self.config.get('max_train_prompts', ''))
        ttk.Entry(frame, textvariable=self.max_prompts_var, width=20).grid(row=1, column=1, sticky=W, padx=5, pady=2)
        
        # Resolution
        ttk.Label(frame, text="Resolution:").grid(row=2, column=0, sticky=W, padx=5, pady=2)
        self.resolution_var = IntVar(value=self.config['resolution'])
        resolution_combo = ttk.Combobox(frame, textvariable=self.resolution_var, values=[512, 768, 1024], width=18)
        resolution_combo.grid(row=2, column=1, sticky=W, padx=5, pady=2)
        
        return row + 1
    
    def setup_training_section(self, parent, row):
        """Setup training parameters section."""
        frame = ttk.LabelFrame(parent, text="Training Parameters", style='Section.TLabelframe')
        frame.grid(row=row, column=0, sticky=(W, E), padx=5, pady=5)
        
        # Epochs
        ttk.Label(frame, text="Epochs:").grid(row=0, column=0, sticky=W, padx=5, pady=2)
        self.epochs_var = IntVar(value=self.config['num_epochs'])
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.epochs_var, width=18).grid(row=0, column=1, sticky=W, padx=5, pady=2)
        
        # Batch size
        ttk.Label(frame, text="Batch Size:").grid(row=1, column=0, sticky=W, padx=5, pady=2)
        self.batch_size_var = IntVar(value=self.config['train_batch_size'])
        ttk.Spinbox(frame, from_=1, to=32, textvariable=self.batch_size_var, width=18).grid(row=1, column=1, sticky=W, padx=5, pady=2)
        
        # Learning rate
        ttk.Label(frame, text="Learning Rate:").grid(row=2, column=0, sticky=W, padx=5, pady=2)
        self.lr_var = StringVar(value=str(self.config['learning_rate']))
        ttk.Entry(frame, textvariable=self.lr_var, width=20).grid(row=2, column=1, sticky=W, padx=5, pady=2)
        
        # LR Scheduler
        ttk.Label(frame, text="LR Scheduler:").grid(row=3, column=0, sticky=W, padx=5, pady=2)
        self.lr_scheduler_var = StringVar(value=self.config['lr_scheduler'])
        ttk.Combobox(frame, textvariable=self.lr_scheduler_var, 
                     values=['linear', 'cosine', 'constant', 'constant_with_warmup'],
                     width=18).grid(row=3, column=1, sticky=W, padx=5, pady=2)
        
        # Warmup steps
        ttk.Label(frame, text="Warmup Steps:").grid(row=4, column=0, sticky=W, padx=5, pady=2)
        self.warmup_var = IntVar(value=self.config['lr_warmup_steps'])
        ttk.Spinbox(frame, from_=0, to=5000, textvariable=self.warmup_var, width=18).grid(row=4, column=1, sticky=W, padx=5, pady=2)
        
        return row + 1
    
    def setup_distillation_section(self, parent, row):
        """Setup distillation settings section."""
        frame = ttk.LabelFrame(parent, text="Distillation Settings", style='Section.TLabelframe')
        frame.grid(row=row, column=0, sticky=(W, E), padx=5, pady=5)
        
        # Guidance scale
        ttk.Label(frame, text="Teacher CFG Scale:").grid(row=0, column=0, sticky=W, padx=5, pady=2)
        self.guidance_var = DoubleVar(value=self.config['guidance_scale'])
        ttk.Spinbox(frame, from_=1.0, to=15.0, increment=0.5, textvariable=self.guidance_var, width=18).grid(row=0, column=1, sticky=W, padx=5, pady=2)
        
        # CFG weight
        ttk.Label(frame, text="CFG Weight:").grid(row=1, column=0, sticky=W, padx=5, pady=2)
        self.cfg_weight_var = DoubleVar(value=self.config['cfg_weight'])
        ttk.Spinbox(frame, from_=0.0, to=5.0, increment=0.1, textvariable=self.cfg_weight_var, width=18).grid(row=1, column=1, sticky=W, padx=5, pady=2)
        
        # DM weight
        ttk.Label(frame, text="DM Weight:").grid(row=2, column=0, sticky=W, padx=5, pady=2)
        self.dm_weight_var = DoubleVar(value=self.config['dm_weight'])
        ttk.Spinbox(frame, from_=0.0, to=5.0, increment=0.1, textvariable=self.dm_weight_var, width=18).grid(row=2, column=1, sticky=W, padx=5, pady=2)
        
        # Student inference steps
        ttk.Label(frame, text="Target Steps:").grid(row=3, column=0, sticky=W, padx=5, pady=2)
        self.student_steps_var = IntVar(value=self.config['student_inference_steps'])
        ttk.Spinbox(frame, from_=4, to=50, textvariable=self.student_steps_var, width=18).grid(row=3, column=1, sticky=W, padx=5, pady=2)
        
        # Use LPIPS
        self.use_lpips_var = BooleanVar(value=self.config.get('use_lpips', False))
        ttk.Checkbutton(frame, text="Use LPIPS Perceptual Loss", variable=self.use_lpips_var).grid(row=4, column=0, columnspan=2, sticky=W, padx=5, pady=2)
        
        return row + 1
    
    def setup_lora_section(self, parent, row):
        """Setup LoRA settings section."""
        frame = ttk.LabelFrame(parent, text="LoRA Settings", style='Section.TLabelframe')
        frame.grid(row=row, column=0, sticky=(W, E), padx=5, pady=5)
        
        # Use LoRA
        self.use_lora_var = BooleanVar(value=self.config.get('use_lora', False))
        lora_check = ttk.Checkbutton(frame, text="Enable LoRA Training", variable=self.use_lora_var, command=self.toggle_lora)
        lora_check.grid(row=0, column=0, columnspan=2, sticky=W, padx=5, pady=2)
        
        # LoRA rank
        ttk.Label(frame, text="LoRA Rank:").grid(row=1, column=0, sticky=W, padx=5, pady=2)
        self.lora_rank_var = IntVar(value=self.config.get('lora_rank', 64))
        self.lora_rank_spin = ttk.Spinbox(frame, from_=8, to=256, textvariable=self.lora_rank_var, width=18)
        self.lora_rank_spin.grid(row=1, column=1, sticky=W, padx=5, pady=2)
        
        # LoRA alpha
        ttk.Label(frame, text="LoRA Alpha:").grid(row=2, column=0, sticky=W, padx=5, pady=2)
        self.lora_alpha_var = IntVar(value=self.config.get('lora_alpha', 64))
        self.lora_alpha_spin = ttk.Spinbox(frame, from_=8, to=256, textvariable=self.lora_alpha_var, width=18)
        self.lora_alpha_spin.grid(row=2, column=1, sticky=W, padx=5, pady=2)
        
        self.toggle_lora()
        
        return row + 1
    
    def setup_performance_section(self, parent, row):
        """Setup performance settings section."""
        frame = ttk.LabelFrame(parent, text="Performance & Memory", style='Section.TLabelframe')
        frame.grid(row=row, column=0, sticky=(W, E), padx=5, pady=5)
        
        # BF16
        self.use_bf16_var = BooleanVar(value=self.config.get('use_bf16', True))
        ttk.Checkbutton(frame, text="Use BF16 Precision (Recommended)", variable=self.use_bf16_var).grid(row=0, column=0, columnspan=2, sticky=W, padx=5, pady=2)
        
        # Flash Attention
        self.use_flash_var = BooleanVar(value=self.config.get('use_flash_attention', True))
        ttk.Checkbutton(frame, text="Use Flash Attention (RTX 5090)", variable=self.use_flash_var).grid(row=1, column=0, columnspan=2, sticky=W, padx=5, pady=2)
        
        # Gradient checkpointing
        self.grad_checkpoint_var = BooleanVar(value=self.config.get('gradient_checkpointing', True))
        ttk.Checkbutton(frame, text="Gradient Checkpointing (Save Memory)", variable=self.grad_checkpoint_var).grid(row=2, column=0, columnspan=2, sticky=W, padx=5, pady=2)
        
        return row + 1
    
    def setup_output_section(self, parent, row):
        """Setup output settings section."""
        frame = ttk.LabelFrame(parent, text="Output & Logging", style='Section.TLabelframe')
        frame.grid(row=row, column=0, sticky=(W, E), padx=5, pady=5)
        
        # Output directory
        ttk.Label(frame, text="Output Dir:").grid(row=0, column=0, sticky=W, padx=5, pady=2)
        output_frame = ttk.Frame(frame)
        output_frame.grid(row=0, column=1, sticky=(W, E), padx=5, pady=2)
        
        self.output_dir_var = StringVar(value=self.config['output_dir'])
        ttk.Entry(output_frame, textvariable=self.output_dir_var, width=40).pack(side=LEFT, padx=2)
        ttk.Button(output_frame, text="Browse...", command=self.browse_output_dir).pack(side=LEFT)
        
        # Logging steps
        ttk.Label(frame, text="Log Every N Steps:").grid(row=1, column=0, sticky=W, padx=5, pady=2)
        self.logging_steps_var = IntVar(value=self.config.get('logging_steps', 10))
        ttk.Spinbox(frame, from_=1, to=1000, textvariable=self.logging_steps_var, width=18).grid(row=1, column=1, sticky=W, padx=5, pady=2)
        
        # Save steps
        ttk.Label(frame, text="Save Every N Steps:").grid(row=2, column=0, sticky=W, padx=5, pady=2)
        self.save_steps_var = IntVar(value=self.config.get('save_steps', 1000))
        ttk.Spinbox(frame, from_=100, to=10000, increment=100, textvariable=self.save_steps_var, width=18).grid(row=2, column=1, sticky=W, padx=5, pady=2)
        
        # Validation steps
        ttk.Label(frame, text="Validate Every N Steps:").grid(row=3, column=0, sticky=W, padx=5, pady=2)
        self.validation_steps_var = IntVar(value=self.config.get('validation_steps', 500))
        ttk.Spinbox(frame, from_=100, to=10000, increment=100, textvariable=self.validation_steps_var, width=18).grid(row=3, column=1, sticky=W, padx=5, pady=2)
        
        return row + 1
    
    def setup_control_buttons(self, parent, row):
        """Setup control buttons."""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky=(W, E), padx=5, pady=10)
        
        # Start button
        self.start_btn = ttk.Button(frame, text="▶ Start Training", command=self.start_training, style='Success.TButton')
        self.start_btn.pack(side=LEFT, padx=5)
        
        # Stop button
        self.stop_btn = ttk.Button(frame, text="⏹ Stop Training", command=self.stop_training, state=DISABLED, style='Danger.TButton')
        self.stop_btn.pack(side=LEFT, padx=5)
        
        # Test model button
        ttk.Button(frame, text="🧪 Test Model", command=self.test_model).pack(side=LEFT, padx=5)
    
    def setup_monitoring_panel(self, parent):
        """Setup monitoring panel."""
        # Notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=BOTH, expand=True)
        
        # Progress tab
        progress_frame = ttk.Frame(notebook)
        notebook.add(progress_frame, text="Progress")
        self.setup_progress_tab(progress_frame)
        
        # Logs tab
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Logs")
        self.setup_logs_tab(logs_frame)
        
        # Validation tab
        validation_frame = ttk.Frame(notebook)
        notebook.add(validation_frame, text="Validation")
        self.setup_validation_tab(validation_frame)
        
        # System tab
        system_frame = ttk.Frame(notebook)
        notebook.add(system_frame, text="System")
        self.setup_system_tab(system_frame)
    
    def setup_progress_tab(self, parent):
        """Setup progress monitoring tab."""
        # Progress info
        info_frame = ttk.LabelFrame(parent, text="Training Progress")
        info_frame.pack(fill=X, padx=5, pady=5)
        
        # Current epoch
        ttk.Label(info_frame, text="Current Epoch:").grid(row=0, column=0, sticky=W, padx=5, pady=2)
        self.epoch_label = ttk.Label(info_frame, text="0 / 0")
        self.epoch_label.grid(row=0, column=1, sticky=W, padx=5, pady=2)
        
        # Current step
        ttk.Label(info_frame, text="Current Step:").grid(row=1, column=0, sticky=W, padx=5, pady=2)
        self.step_label = ttk.Label(info_frame, text="0 / 0")
        self.step_label.grid(row=1, column=1, sticky=W, padx=5, pady=2)
        
        # Progress bar
        self.progress_var = DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(info_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=2, sticky=(W, E), padx=5, pady=5)
        
        # Loss metrics
        metrics_frame = ttk.LabelFrame(parent, text="Loss Metrics")
        metrics_frame.pack(fill=X, padx=5, pady=5)
        
        # Total loss
        ttk.Label(metrics_frame, text="Total Loss:").grid(row=0, column=0, sticky=W, padx=5, pady=2)
        self.total_loss_label = ttk.Label(metrics_frame, text="N/A")
        self.total_loss_label.grid(row=0, column=1, sticky=W, padx=5, pady=2)
        
        # CFG loss
        ttk.Label(metrics_frame, text="CFG Loss:").grid(row=1, column=0, sticky=W, padx=5, pady=2)
        self.cfg_loss_label = ttk.Label(metrics_frame, text="N/A")
        self.cfg_loss_label.grid(row=1, column=1, sticky=W, padx=5, pady=2)
        
        # DM loss
        ttk.Label(metrics_frame, text="DM Loss:").grid(row=2, column=0, sticky=W, padx=5, pady=2)
        self.dm_loss_label = ttk.Label(metrics_frame, text="N/A")
        self.dm_loss_label.grid(row=2, column=1, sticky=W, padx=5, pady=2)
        
        # Learning rate
        ttk.Label(metrics_frame, text="Learning Rate:").grid(row=3, column=0, sticky=W, padx=5, pady=2)
        self.lr_label = ttk.Label(metrics_frame, text="N/A")
        self.lr_label.grid(row=3, column=1, sticky=W, padx=5, pady=2)
        
        # Time estimate
        time_frame = ttk.LabelFrame(parent, text="Time Estimate")
        time_frame.pack(fill=X, padx=5, pady=5)
        
        ttk.Label(time_frame, text="Elapsed:").grid(row=0, column=0, sticky=W, padx=5, pady=2)
        self.elapsed_label = ttk.Label(time_frame, text="00:00:00")
        self.elapsed_label.grid(row=0, column=1, sticky=W, padx=5, pady=2)
        
        ttk.Label(time_frame, text="Remaining:").grid(row=1, column=0, sticky=W, padx=5, pady=2)
        self.remaining_label = ttk.Label(time_frame, text="N/A")
        self.remaining_label.grid(row=1, column=1, sticky=W, padx=5, pady=2)
    
    def setup_logs_tab(self, parent):
        """Setup logs tab."""
        # Log display
        self.log_text = scrolledtext.ScrolledText(parent, wrap=WORD, height=30)
        self.log_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Clear button
        ttk.Button(parent, text="Clear Logs", command=self.clear_logs).pack(pady=5)
    
    def setup_validation_tab(self, parent):
        """Setup validation images tab."""
        if not PIL_AVAILABLE:
            ttk.Label(parent, text="PIL not installed. Image preview disabled.").pack(pady=20)
            return
        
        # Image display
        self.validation_canvas = Canvas(parent, bg='gray20')
        self.validation_canvas.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Controls
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=X, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="◀ Previous", command=self.prev_validation_image).pack(side=LEFT, padx=2)
        ttk.Button(controls_frame, text="▶ Next", command=self.next_validation_image).pack(side=LEFT, padx=2)
        ttk.Button(controls_frame, text="🔄 Refresh", command=self.refresh_validation_images).pack(side=LEFT, padx=2)
        
        self.validation_index = 0
        self.validation_images = []
    
    def setup_system_tab(self, parent):
        """Setup system monitoring tab."""
        # GPU info
        gpu_frame = ttk.LabelFrame(parent, text="GPU Information")
        gpu_frame.pack(fill=X, padx=5, pady=5)
        
        ttk.Label(gpu_frame, text="GPU:").grid(row=0, column=0, sticky=W, padx=5, pady=2)
        self.gpu_name_label = ttk.Label(gpu_frame, text="Detecting...")
        self.gpu_name_label.grid(row=0, column=1, sticky=W, padx=5, pady=2)
        
        ttk.Label(gpu_frame, text="VRAM Usage:").grid(row=1, column=0, sticky=W, padx=5, pady=2)
        self.vram_label = ttk.Label(gpu_frame, text="0 GB / 0 GB")
        self.vram_label.grid(row=1, column=1, sticky=W, padx=5, pady=2)
        
        self.vram_progress = ttk.Progressbar(gpu_frame, maximum=100)
        self.vram_progress.grid(row=2, column=0, columnspan=2, sticky=(W, E), padx=5, pady=2)
        
        # System info
        sys_frame = ttk.LabelFrame(parent, text="System Information")
        sys_frame.pack(fill=X, padx=5, pady=5)
        
        ttk.Label(sys_frame, text="CPU Usage:").grid(row=0, column=0, sticky=W, padx=5, pady=2)
        self.cpu_label = ttk.Label(sys_frame, text="0%")
        self.cpu_label.grid(row=0, column=1, sticky=W, padx=5, pady=2)
        
        ttk.Label(sys_frame, text="RAM Usage:").grid(row=1, column=0, sticky=W, padx=5, pady=2)
        self.ram_label = ttk.Label(sys_frame, text="0 GB / 0 GB")
        self.ram_label.grid(row=1, column=1, sticky=W, padx=5, pady=2)
        
        self.detect_gpu()
    
    def setup_status_bar(self):
        """Setup status bar."""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=SUNKEN, anchor=W)
        self.status_bar.pack(side=BOTTOM, fill=X)
    
    # Configuration methods
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'teacher_model': 'Tongyi-MAI/Z-Image-Base',
            'student_model': '',
            'train_data_file': 'sample_training_data.json',
            'max_train_prompts': '',
            'resolution': 1024,
            'num_epochs': 10,
            'train_batch_size': 4,
            'learning_rate': 1e-5,
            'lr_scheduler': 'cosine',
            'lr_warmup_steps': 500,
            'guidance_scale': 7.5,
            'cfg_weight': 1.0,
            'dm_weight': 0.5,
            'student_inference_steps': 8,
            'use_lpips': False,
            'use_lora': False,
            'lora_rank': 64,
            'lora_alpha': 64,
            'use_bf16': True,
            'use_flash_attention': True,
            'gradient_checkpointing': True,
            'output_dir': './distillation_output',
            'logging_steps': 10,
            'save_steps': 1000,
            'validation_steps': 500,
        }
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration from UI."""
        config = {
            'teacher_model': self.teacher_model_var.get(),
            'train_data_file': self.train_data_var.get(),
            'output_dir': self.output_dir_var.get(),
            'resolution': self.resolution_var.get(),
            'num_epochs': self.epochs_var.get(),
            'train_batch_size': self.batch_size_var.get(),
            'learning_rate': float(self.lr_var.get()),
            'lr_scheduler': self.lr_scheduler_var.get(),
            'lr_warmup_steps': self.warmup_var.get(),
            'guidance_scale': self.guidance_var.get(),
            'cfg_weight': self.cfg_weight_var.get(),
            'dm_weight': self.dm_weight_var.get(),
            'student_inference_steps': self.student_steps_var.get(),
            'use_lpips': self.use_lpips_var.get(),
            'use_lora': self.use_lora_var.get(),
            'lora_rank': self.lora_rank_var.get(),
            'lora_alpha': self.lora_alpha_var.get(),
            'use_bf16': self.use_bf16_var.get(),
            'use_flash_attention': self.use_flash_var.get(),
            'gradient_checkpointing': self.grad_checkpoint_var.get(),
            'logging_steps': self.logging_steps_var.get(),
            'save_steps': self.save_steps_var.get(),
            'validation_steps': self.validation_steps_var.get(),
        }
        
        if self.student_model_var.get():
            config['student_model'] = self.student_model_var.get()
        
        if self.max_prompts_var.get():
            config['max_train_prompts'] = int(self.max_prompts_var.get())
        
        return config
    
    def new_config(self):
        """Create new configuration."""
        if messagebox.askyesno("New Configuration", "Reset to default configuration?"):
            self.config = self.load_default_config()
            self.apply_config(self.config)
    
    def load_config(self):
        """Load configuration from file."""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                self.config = config
                self.apply_config(config)
                messagebox.showinfo("Success", "Configuration loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def save_config(self):
        """Save configuration to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                config = self.get_current_config()
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                messagebox.showinfo("Success", "Configuration saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def apply_config(self, config: Dict[str, Any]):
        """Apply configuration to UI."""
        self.teacher_model_var.set(config.get('teacher_model', ''))
        self.student_model_var.set(config.get('student_model', ''))
        self.train_data_var.set(config.get('train_data_file', ''))
        self.max_prompts_var.set(config.get('max_train_prompts', ''))
        self.resolution_var.set(config.get('resolution', 1024))
        self.epochs_var.set(config.get('num_epochs', 10))
        self.batch_size_var.set(config.get('train_batch_size', 4))
        self.lr_var.set(str(config.get('learning_rate', 1e-5)))
        self.lr_scheduler_var.set(config.get('lr_scheduler', 'cosine'))
        self.warmup_var.set(config.get('lr_warmup_steps', 500))
        self.guidance_var.set(config.get('guidance_scale', 7.5))
        self.cfg_weight_var.set(config.get('cfg_weight', 1.0))
        self.dm_weight_var.set(config.get('dm_weight', 0.5))
        self.student_steps_var.set(config.get('student_inference_steps', 8))
        self.use_lpips_var.set(config.get('use_lpips', False))
        self.use_lora_var.set(config.get('use_lora', False))
        self.lora_rank_var.set(config.get('lora_rank', 64))
        self.lora_alpha_var.set(config.get('lora_alpha', 64))
        self.use_bf16_var.set(config.get('use_bf16', True))
        self.use_flash_var.set(config.get('use_flash_attention', True))
        self.grad_checkpoint_var.set(config.get('gradient_checkpointing', True))
        self.output_dir_var.set(config.get('output_dir', './distillation_output'))
        self.logging_steps_var.set(config.get('logging_steps', 10))
        self.save_steps_var.set(config.get('save_steps', 1000))
        self.validation_steps_var.set(config.get('validation_steps', 500))
        
        self.toggle_lora()
    
    def apply_preset(self, preset_name: str):
        """Apply preset configuration."""
        presets = {
            'full': {
                'train_batch_size': 4,
                'learning_rate': 1e-5,
                'use_lora': False,
                'num_epochs': 10,
            },
            'lora': {
                'train_batch_size': 8,
                'learning_rate': 5e-5,
                'use_lora': True,
                'lora_rank': 64,
                'lora_alpha': 64,
                'num_epochs': 5,
            },
            'test': {
                'train_batch_size': 8,
                'learning_rate': 5e-5,
                'use_lora': True,
                'lora_rank': 32,
                'num_epochs': 3,
                'max_train_prompts': 100,
            },
            'production': {
                'train_batch_size': 4,
                'learning_rate': 5e-6,
                'use_lora': False,
                'use_lpips': True,
                'dm_weight': 1.0,
                'num_epochs': 20,
            },
        }
        
        if preset_name in presets:
            config = {**self.config, **presets[preset_name]}
            self.apply_config(config)
            messagebox.showinfo("Preset Applied", f"Applied {preset_name.title()} preset configuration.")
    
    # UI callbacks
    
    def toggle_lora(self):
        """Toggle LoRA settings visibility."""
        state = NORMAL if self.use_lora_var.get() else DISABLED
        self.lora_rank_spin.config(state=state)
        self.lora_alpha_spin.config(state=state)
    
    def browse_train_data(self):
        """Browse for training data file."""
        filename = filedialog.askopenfilename(
            title="Select Training Data",
            filetypes=[("JSON files", "*.json"), ("JSONL files", "*.jsonl"), ("All files", "*.*")]
        )
        if filename:
            self.train_data_var.set(filename)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
    
    # Training control
    
    def start_training(self):
        """Start training process."""
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already running!")
            return
        
        # Validate configuration
        if not self.validate_config():
            return
        
        # Build command
        config = self.get_current_config()
        cmd = self.build_training_command(config)
        
        # Confirm start
        if not messagebox.askyesno("Start Training", "Start distillation training with current configuration?"):
            return
        
        # Start training in thread
        self.is_training = True
        self.start_btn.config(state=DISABLED)
        self.stop_btn.config(state=NORMAL)
        self.update_status("Training started...")
        
        self.training_thread = threading.Thread(target=self.run_training, args=(cmd,), daemon=True)
        self.training_thread.start()
        
        # Start log monitoring
        self.monitor_training()
    
    def stop_training(self):
        """Stop training process."""
        if not self.is_training:
            return
        
        if messagebox.askyesno("Stop Training", "Are you sure you want to stop training?"):
            if self.training_process:
                self.training_process.terminate()
            self.is_training = False
            self.start_btn.config(state=NORMAL)
            self.stop_btn.config(state=DISABLED)
            self.update_status("Training stopped.")
            self.log("Training stopped by user.\n")
    
    def run_training(self, cmd: list):
        """Run training process."""
        try:
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Read output
            for line in self.training_process.stdout:
                if not self.is_training:
                    break
                self.log(line)
                self.parse_training_output(line)
            
            self.training_process.wait()
            
            if self.training_process.returncode == 0:
                self.update_status("Training completed successfully!")
                messagebox.showinfo("Success", "Training completed successfully!")
            else:
                self.update_status("Training failed.")
                messagebox.showerror("Error", "Training failed. Check logs for details.")
        
        except Exception as e:
            self.log(f"Error: {e}\n")
            self.update_status("Training error.")
            messagebox.showerror("Error", f"Training error: {e}")
        
        finally:
            self.is_training = False
            self.start_btn.config(state=NORMAL)
            self.stop_btn.config(state=DISABLED)
            self.training_process = None
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        # Check training data file
        if not self.train_data_var.get():
            messagebox.showerror("Error", "Please select training data file!")
            return False
        
        if not os.path.exists(self.train_data_var.get()):
            messagebox.showerror("Error", "Training data file does not exist!")
            return False
        
        # Check output directory
        if not self.output_dir_var.get():
            messagebox.showerror("Error", "Please specify output directory!")
            return False
        
        return True
    
    def build_training_command(self, config: Dict[str, Any]) -> list:
        """Build training command from configuration."""
        cmd = ['python', 'z_image_distillation_trainer.py']
        
        # Required arguments
        cmd.extend(['--teacher_model', config['teacher_model']])
        cmd.extend(['--train_data_file', config['train_data_file']])
        cmd.extend(['--output_dir', config['output_dir']])
        
        # Optional arguments
        if 'student_model' in config and config['student_model']:
            cmd.extend(['--student_model', config['student_model']])
        
        if 'max_train_prompts' in config:
            cmd.extend(['--max_train_prompts', str(config['max_train_prompts'])])
        
        cmd.extend(['--resolution', str(config['resolution'])])
        cmd.extend(['--num_epochs', str(config['num_epochs'])])
        cmd.extend(['--train_batch_size', str(config['train_batch_size'])])
        cmd.extend(['--learning_rate', str(config['learning_rate'])])
        cmd.extend(['--lr_scheduler', config['lr_scheduler']])
        cmd.extend(['--lr_warmup_steps', str(config['lr_warmup_steps'])])
        cmd.extend(['--guidance_scale', str(config['guidance_scale'])])
        cmd.extend(['--cfg_weight', str(config['cfg_weight'])])
        cmd.extend(['--dm_weight', str(config['dm_weight'])])
        cmd.extend(['--student_inference_steps', str(config['student_inference_steps'])])
        
        if config.get('use_lpips'):
            cmd.append('--use_lpips')
        
        if config.get('use_lora'):
            cmd.append('--use_lora')
            cmd.extend(['--lora_rank', str(config['lora_rank'])])
            cmd.extend(['--lora_alpha', str(config['lora_alpha'])])
        
        if config.get('use_bf16'):
            cmd.append('--use_bf16')
        
        if config.get('use_flash_attention'):
            cmd.append('--use_flash_attention')
        
        if config.get('gradient_checkpointing'):
            cmd.append('--gradient_checkpointing')
        
        cmd.extend(['--logging_steps', str(config['logging_steps'])])
        cmd.extend(['--save_steps', str(config['save_steps'])])
        cmd.extend(['--validation_steps', str(config['validation_steps'])])
        
        return cmd
    
    def test_model(self):
        """Test distilled model."""
        # Find final checkpoint
        output_dir = Path(self.output_dir_var.get())
        final_dir = output_dir / "final"
        
        if not final_dir.exists():
            messagebox.showwarning("Warning", "No trained model found in output directory!")
            return
        
        # Check if LoRA or full model
        is_lora = (final_dir / "lora").exists()
        model_path = str(final_dir / "lora" if is_lora else final_dir / "student")
        
        # Build test command
        cmd = ['python', 'test_distilled_model.py', '--model_path', model_path]
        
        if is_lora:
            cmd.append('--is_lora')
        
        # Run test script
        try:
            subprocess.Popen(cmd)
            self.update_status("Launched test script...")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch test script: {e}")
    
    # Monitoring methods
    
    def start_monitoring(self):
        """Start monitoring threads."""
        self.update_system_info()
    
    def update_system_info(self):
        """Update system information."""
        if PSUTIL_AVAILABLE:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_label.config(text=f"{cpu_percent:.1f}%")
            
            # RAM usage
            ram = psutil.virtual_memory()
            ram_used = ram.used / (1024**3)
            ram_total = ram.total / (1024**3)
            self.ram_label.config(text=f"{ram_used:.1f} GB / {ram_total:.1f} GB")
        
        # Schedule next update
        self.root.after(2000, self.update_system_info)
    
    def detect_gpu(self):
        """Detect GPU information."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.gpu_name_label.config(text=gpu_name)
                self.update_vram()
        except Exception as e:
            self.gpu_name_label.config(text="N/A")
    
    def update_vram(self):
        """Update VRAM usage."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(0) / (1024**3)
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.vram_label.config(text=f"{vram_used:.1f} GB / {vram_total:.1f} GB")
                
                vram_percent = (vram_used / vram_total) * 100
                self.vram_progress['value'] = vram_percent
        except Exception:
            pass
        
        # Schedule next update
        if self.is_training:
            self.root.after(2000, self.update_vram)
    
    def monitor_training(self):
        """Monitor training progress."""
        if self.is_training:
            self.update_vram()
            self.root.after(1000, self.monitor_training)
    
    def parse_training_output(self, line: str):
        """Parse training output for metrics."""
        # Parse epoch and step info
        if "Epoch" in line and "Step" in line:
            # Extract metrics if available
            if "loss=" in line:
                try:
                    # Example: "Epoch 5, Step 1000: loss=0.0234, cfg_loss=0.0156, dm_loss=0.0078, lr=5.00e-06"
                    parts = line.split(":")
                    if len(parts) >= 2:
                        epoch_step = parts[0]
                        metrics = parts[1]
                        
                        # Update epoch/step labels
                        self.update_status(epoch_step.strip())
                        
                        # Parse losses
                        if "loss=" in metrics:
                            total_loss = float(metrics.split("loss=")[1].split(",")[0])
                            self.total_loss_label.config(text=f"{total_loss:.4f}")
                        
                        if "cfg_loss=" in metrics:
                            cfg_loss = float(metrics.split("cfg_loss=")[1].split(",")[0])
                            self.cfg_loss_label.config(text=f"{cfg_loss:.4f}")
                        
                        if "dm_loss=" in metrics:
                            dm_loss = float(metrics.split("dm_loss=")[1].split(",")[0])
                            self.dm_loss_label.config(text=f"{dm_loss:.4f}")
                        
                        if "lr=" in metrics:
                            lr = metrics.split("lr=")[1].split()[0]
                            self.lr_label.config(text=lr)
                except Exception:
                    pass
    
    def log(self, message: str):
        """Add message to log."""
        self.log_text.insert(END, message)
        self.log_text.see(END)
    
    def clear_logs(self):
        """Clear log display."""
        self.log_text.delete(1.0, END)
    
    def update_status(self, message: str):
        """Update status bar."""
        self.status_bar.config(text=message)
    
    # Validation images
    
    def refresh_validation_images(self):
        """Refresh validation images list."""
        if not PIL_AVAILABLE:
            return
        
        output_dir = Path(self.output_dir_var.get())
        validation_dir = output_dir / "validation"
        
        if not validation_dir.exists():
            return
        
        # Find all validation images
        self.validation_images = sorted(validation_dir.rglob("*.png"))
        self.validation_index = 0
        
        if self.validation_images:
            self.show_validation_image()
    
    def show_validation_image(self):
        """Show current validation image."""
        if not PIL_AVAILABLE or not self.validation_images:
            return
        
        if 0 <= self.validation_index < len(self.validation_images):
            img_path = self.validation_images[self.validation_index]
            
            try:
                img = Image.open(img_path)
                
                # Resize to fit canvas
                canvas_width = self.validation_canvas.winfo_width()
                canvas_height = self.validation_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(img)
                
                self.validation_canvas.delete("all")
                self.validation_canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    image=photo,
                    anchor=CENTER
                )
                self.validation_canvas.image = photo  # Keep reference
                
            except Exception as e:
                self.log(f"Error loading validation image: {e}\n")
    
    def prev_validation_image(self):
        """Show previous validation image."""
        if self.validation_images and self.validation_index > 0:
            self.validation_index -= 1
            self.show_validation_image()
    
    def next_validation_image(self):
        """Show next validation image."""
        if self.validation_images and self.validation_index < len(self.validation_images) - 1:
            self.validation_index += 1
            self.show_validation_image()
    
    # Help dialogs
    
    def show_docs(self):
        """Show documentation."""
        messagebox.showinfo(
            "Documentation",
            "Z-Image Distillation Trainer\n\n"
            "For detailed documentation, see README.md\n\n"
            "Key Features:\n"
            "- Decoupled-DMD distillation algorithm\n"
            "- Full fine-tuning and LoRA training\n"
            "- RTX 5090 optimized\n"
            "- Real-time monitoring\n"
            "- Validation image preview\n\n"
            "Quick Start:\n"
            "1. Configure training parameters\n"
            "2. Select training data file\n"
            "3. Click 'Start Training'\n"
            "4. Monitor progress in real-time"
        )
    
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About",
            "Z-Image Distillation Trainer GUI\n\n"
            "Version: 1.0\n"
            "Author: marduk191\n"
            "License: Apache 2.0\n\n"
            "Based on:\n"
            "- Decoupled-DMD (arXiv:2511.22677)\n"
            "- Z-Image (arXiv:2511.22699)\n\n"
            "GitHub: Tongyi-MAI/Z-Image"
        )
    
    def on_closing(self):
        """Handle window closing."""
        if self.is_training:
            if messagebox.askyesno("Training in Progress", "Training is in progress. Stop training and exit?"):
                if self.training_process:
                    self.training_process.terminate()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """Main entry point."""
    root = Tk()
    app = ZImageDistillationGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
