#!/usr/bin/env python3
"""
Test script for distilled Z-Image models.
Generates sample images to verify distillation quality.
"""

import argparse
import torch
from pathlib import Path
from diffusers import ZImagePipeline
from peft import PeftModel


def test_distilled_model(args):
    """Test the distilled model with sample prompts."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    
    if args.is_lora:
        print("Loading as LoRA model...")
        # Load base model
        pipe = ZImagePipeline.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
        ).to(device)
        
        # Load LoRA weights
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            args.model_path,
        )
    else:
        print("Loading as full model...")
        pipe = ZImagePipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
        ).to(device)
    
    print("Model loaded successfully!")
    
    # Test prompts
    test_prompts = [
        "A serene mountain landscape at golden hour, photorealistic",
        "Portrait of an elderly woman, kind eyes, natural lighting",
        "Futuristic cyberpunk city at night, neon lights, rain",
        "Close-up macro photography of a dewdrop on rose petal",
        "Ancient temple ruins overgrown with vegetation",
    ]
    
    if args.prompt:
        test_prompts = [args.prompt]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {len(test_prompts)} test images...")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print("")
    
    # Generate images
    for idx, prompt in enumerate(test_prompts):
        print(f"[{idx+1}/{len(test_prompts)}] Generating: {prompt[:50]}...")
        
        try:
            image = pipe(
                prompt=prompt,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device=device).manual_seed(args.seed + idx),
            ).images[0]
            
            # Save image
            output_path = output_dir / f"test_{idx:03d}.png"
            image.save(output_path)
            
            # Save prompt
            prompt_path = output_dir / f"test_{idx:03d}.txt"
            with open(prompt_path, 'w') as f:
                f.write(prompt)
            
            print(f"  ✓ Saved to: {output_path}")
        
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"\n✓ Test complete! Images saved to: {output_dir}")
    print("\nInspect the images to verify distillation quality:")
    print("- Image clarity and detail")
    print("- Prompt adherence")
    print("- Color accuracy")
    print("- Lack of artifacts")


def main():
    parser = argparse.ArgumentParser(description="Test distilled Z-Image model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to distilled model checkpoint",
    )
    parser.add_argument(
        "--is_lora",
        action="store_true",
        help="Model is a LoRA checkpoint",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo",
        help="Base model for LoRA (only used if --is_lora)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_outputs",
        help="Directory to save test images",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt to test (overrides default prompts)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=8,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale (typically 1.0 for distilled models)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution (square)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision",
    )
    
    args = parser.parse_args()
    test_distilled_model(args)


if __name__ == "__main__":
    main()
