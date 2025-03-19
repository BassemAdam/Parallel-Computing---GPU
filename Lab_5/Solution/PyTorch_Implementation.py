import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import argparse
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

class RGBToGrayscaleConvolution:
    def __init__(self, mask_file, device='cuda'):
        """
        Initialize RGB to grayscale convolution with a specified mask.
        
        Args:
            mask_file (str): Path to the mask file
            device (str): Device to run the convolution on ('cuda' or 'cpu')
        """
        self.device = device
        self.mask = self._load_mask(mask_file)
        self.kernel_size = self.mask.shape[0]
        
        # Create a 2D convolutional layer that operates on all three channels together
        # to produce a single grayscale output channel
        self.conv = nn.Conv2d(
            in_channels=3,   # RGB input
            out_channels=1,  # Grayscale output
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,  # Same padding
            bias=False       # No bias term needed
        )
        
        # Set the weights to our mask values and make them non-learnable
        with torch.no_grad():
            # Create a 4D tensor [out_channels, in_channels, height, width]
            # Apply the same mask to each RGB channel
            mask_tensor = torch.tensor(self.mask, dtype=torch.float32)
            mask_tensor = mask_tensor.unsqueeze(0).repeat(3, 1, 1)
            mask_tensor = mask_tensor.unsqueeze(0)  # [1, 3, kernel_size, kernel_size]
            self.conv.weight = nn.Parameter(mask_tensor, requires_grad=False)
        
        # Move the model to the specified device
        self.conv = self.conv.to(device)
        
        print(f"Initialized RGB to Grayscale convolution with {self.kernel_size}x{self.kernel_size} mask")
        print(f"Using device: {device}")
        
    def _load_mask(self, mask_file):
        """
        Load a convolution mask from file.
        
        Args:
            mask_file (str): Path to the mask file
            
        Returns:
            numpy.ndarray: The loaded mask as a 2D array
        """
        with open(mask_file, 'r') as f:
            # Read mask size
            mask_size = int(f.readline().strip())
            
            # Read mask values
            mask = []
            for _ in range(mask_size):
                row = list(map(float, f.readline().strip().split()))
                mask.append(row)
                
        return np.array(mask)
    
    def apply_convolution(self, images, stride=1):
        """
        Apply the convolution to a batch of images and normalize results.
        
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W]
            stride (int): Stride value for convolution
            
        Returns:
            torch.Tensor: Batch of convolved grayscale images [B, 1, H, W]
        """
      # If stride is 1, we can use the pre-configured conv layer
        if stride == 1:
            with torch.no_grad():
                output = self.conv(images)
        else:
            # For other strides, create a new temporary convolution
            with torch.no_grad():
                temp_conv = nn.Conv2d(
                    in_channels=3,
                    out_channels=1,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    padding=self.kernel_size // 2,
                    bias=False
                ).to(self.device)
                
                # Set the weights to our mask values
                temp_conv.weight = self.conv.weight
                
                # Apply convolution
                output = temp_conv(images)
                print(f"Input shape: {images.shape}, Output shape: {output.shape}")

        # Print a sample of raw output values
        print(f"\nRaw convolution output (sample):")
        print(output[0, 0, 10:15, 10:15])  # Print a 5x5 patch from first image
        print(f"Raw output min: {torch.min(output).item()}, max: {torch.max(output).item()}")
        
        max_values = torch.amax(torch.abs(output), dim=(1, 2, 3), keepdim=True)
        normalized = output / (3.0)
        normalized = torch.where(normalized < 0, torch.zeros_like(normalized), normalized)
        # Normalize by the max value (avoid division by zero)
        max_values = torch.clamp(max_values, min=1e-8)  # Avoid division by zero
        normalized = output / max_values
        

        return normalized

    def process_images(self, input_folder, output_folder, batch_size=1, stride=1):
        """
        Process all images in the input folder and save results to the output folder.
        
        Args:
            input_folder (str): Path to the input folder
            output_folder (str): Path to the output folder
            batch_size (int): Number of images to process in a batch
            stride (int): Stride value for convolution
            
        Returns:
            float: Total processing time in seconds
        """
        # Make sure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # List all image files in the input folder
        image_files = [f for f in os.listdir(input_folder) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(image_files)} images in {input_folder}")
        
        # Image transformation pipeline
        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Force resize to exactly 512x512
            transforms.ToTensor(),          # Convert to tensor [0,1]
        ])
        
        # Process images in batches
        total_images = len(image_files)
        start_time = time.time()
        
        for i in range(0, total_images, batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_images = []
            
            # Load images in this batch
            for img_file in batch_files:
                img_path = os.path.join(input_folder, img_file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    tensor_img = transform(img)
                    batch_images.append(tensor_img)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            # Skip if batch is empty
            if not batch_images:
                continue
            
            # Stack images into a batch tensor
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Apply convolution
            print(f"Processing batch {i//batch_size + 1}/{(total_images+batch_size-1)//batch_size} " + 
                 f"({len(batch_images)} images)")
            
            output_tensor = self.apply_convolution(batch_tensor, stride)
            
            # Save output images
            for j, img_file in enumerate(batch_files[:len(batch_images)]):
                # Get grayscale output and convert to numpy
                output_img = output_tensor[j].cpu().squeeze(0).numpy()
                
                # Convert to uint8 range [0, 255]
                output_img = (output_img * 255).astype(np.uint8)
                
                # Save the output image as grayscale
                output_path = os.path.join(output_folder, img_file)
                Image.fromarray(output_img, mode='L').save(output_path)
                
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Processed {total_images} images in {total_time:.4f} seconds " +
              f"({total_images/total_time:.2f} images/second)")
        
        return total_time


def main():
    parser = argparse.ArgumentParser(description='PyTorch RGB to Grayscale Convolution')
    parser.add_argument('input_folder', help='Path to input image folder')
    parser.add_argument('output_folder', help='Path to output image folder')
    parser.add_argument('batch_size', type=int, help='Batch size for processing')
    parser.add_argument('mask_file', help='Path to convolution mask file')
    parser.add_argument('--stride', type=int, default=1, help='Stride value for convolution (default: 1)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of CUDA')
    
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    
    # Print configuration
    print("Configuration:")
    print(f"  Input folder: {args.input_folder}")
    print(f"  Output folder: {args.output_folder}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Mask file: {args.mask_file}")
    print(f"  Stride: {args.stride}")
    print(f"  Device: {device}")
    
    # Create convolution processor
    processor = RGBToGrayscaleConvolution(args.mask_file, device=device)
    
    # Process images
    processing_time = processor.process_images(
        args.input_folder, 
        args.output_folder, 
        args.batch_size, 
        args.stride
    )
    
    print(f"Total processing time: {processing_time:.4f} seconds")


if __name__ == "__main__":
    main()