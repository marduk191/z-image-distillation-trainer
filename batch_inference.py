#!/usr/bin/env python3
"""
Batch inference utility for distilled Z-Image models.
Efficiently generates images from a prompt list for quality assessment.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

import torch
from diffusers import ZImagePipeline
from peft import PeftModel
from tqdm import tqdm


class BatchInference:
    """Batch inference for distilled models."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        print(f"Loading model from: {args.model_path}")
        
        # Load model
        if args.is_lora:
            self.pipe = ZImagePipeline.from_pretrained(
                args.base_model,
                torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
            ).to(self.device)
            
            self.pipe.transformer = PeftModel.from_pretrained(
                self.pipe.transformer,
                args.model_path,
            )
        else:
            self.pipe = ZImagePipeline.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
            ).to(self.device)
        
        # Enable optimizations
        if args.compile:
            print("Compiling model for faster inference...")
            self.pipe.transformer.compile()
        
        print("Model loaded successfully!")
    
    def load_prompts(self, prompt_file: str) -> List[Dict]:
        """Load prompts from JSON/JSONL file."""
        prompts = []
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            if prompt_file.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        prompts.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    # List of strings or dicts
                    for item in data:
                        if isinstance(item, str):
                            prompts.append({'prompt': item, 'negative_prompt': ''})
                        else:
                            prompts.append(item)
                elif isinstance(data, dict) and 'prompts' in data:
                    prompts = data['prompts']
        
        return prompts
    
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        negative_prompts: List[str],
        batch_idx: int,
    ) -> List:
        """Generate a batch of images."""
        images = []
        
        for i, (prompt, negative_prompt) in enumerate(zip(prompts, negative_prompts)):
            seed = self.args.seed + batch_idx * len(prompts) + i
            
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=self.args.resolution,
                width=self.args.resolution,
                num_inference_steps=self.args.num_inference_steps,
                guidance_scale=self.args.guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            ).images[0]
            
            images.append(image)
        
        return images
    
    def run(self):
        """Run batch inference."""
        # Load prompts
        prompts_data = self.load_prompts(self.args.prompt_file)
        
        if self.args.max_prompts:
            prompts_data = prompts_data[:self.args.max_prompts]
        
        print(f"Loaded {len(prompts_data)} prompts")
        
        # Create output directory
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Batch processing
        total_batches = (len(prompts_data) + self.args.batch_size - 1) // self.args.batch_size
        
        print(f"\nGenerating images:")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Total batches: {total_batches}")
        print(f"  Inference steps: {self.args.num_inference_steps}")
        print(f"  Resolution: {self.args.resolution}x{self.args.resolution}")
        print("")
        
        total_time = 0
        successful = 0
        failed = 0
        
        with tqdm(total=len(prompts_data), desc="Generating") as pbar:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.args.batch_size
                end_idx = min(start_idx + self.args.batch_size, len(prompts_data))
                
                batch_data = prompts_data[start_idx:end_idx]
                
                # Extract prompts and negative prompts
                prompts = []
                negative_prompts = []
                
                for item in batch_data:
                    if isinstance(item, str):
                        prompts.append(item)
                        negative_prompts.append("")
                    else:
                        prompts.append(item.get('prompt', ''))
                        negative_prompts.append(item.get('negative_prompt', ''))
                
                # Generate images
                try:
                    start_time = time.time()
                    images = self.generate_batch(prompts, negative_prompts, batch_idx)
                    batch_time = time.time() - start_time
                    total_time += batch_time
                    
                    # Save images
                    for i, (image, prompt_data) in enumerate(zip(images, batch_data)):
                        img_idx = start_idx + i
                        
                        # Save image
                        image_path = output_dir / f"{img_idx:05d}.png"
                        image.save(image_path)
                        
                        # Save metadata
                        if self.args.save_metadata:
                            metadata_path = output_dir / f"{img_idx:05d}.json"
                            metadata = {
                                'prompt': prompts[i],
                                'negative_prompt': negative_prompts[i],
                                'seed': self.args.seed + batch_idx * self.args.batch_size + i,
                                'num_inference_steps': self.args.num_inference_steps,
                                'guidance_scale': self.args.guidance_scale,
                                'resolution': self.args.resolution,
                            }
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2)
                    
                    successful += len(images)
                    
                except Exception as e:
                    print(f"\nError in batch {batch_idx}: {e}")
                    failed += len(prompts)
                
                pbar.update(len(batch_data))
        
        # Summary
        avg_time = total_time / successful if successful > 0 else 0
        
        print("\n" + "="*60)
        print("Batch Inference Complete")
        print("="*60)
        print(f"Total images: {successful + failed}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per image: {avg_time:.2f}s")
        print(f"Output directory: {output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Batch inference for distilled Z-Image")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to distilled model",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="JSON/JSONL file with prompts",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./batch_outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--is_lora",
        action="store_true",
        help="Model is LoRA checkpoint",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo",
        help="Base model for LoRA",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (process prompts in parallel batches)",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to process",
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
        help="Guidance scale",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        default=True,
        help="Use bfloat16",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model for faster inference",
    )
    parser.add_argument(
        "--save_metadata",
        action="store_true",
        help="Save generation metadata as JSON",
    )
    
    args = parser.parse_args()
    
    inferencer = BatchInference(args)
    inferencer.run()


if __name__ == "__main__":
    main()
