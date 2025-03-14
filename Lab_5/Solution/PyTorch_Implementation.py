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

class RGBConvolution:
    def __init__(self, mask_file, device='cuda'):
        """
        Initialize the RGB convolution with a specified mask.
        
        Args:
            mask_file (str): Path to the mask file
            device (str): Device to run the convolution on ('cuda' or 'cpu')
        """
        self.device = device
        self.mask = self._load_mask(mask_file)
        self.kernel_size = self.mask.shape[0]
        
        # Create a 2D convolutional layer with the fixed mask weights
        self.conv = nn.Conv2d(
            in_channels=3,  # RGB input
            out_channels=3,  # RGB output
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,  # Same padding
            bias=False,      # No bias term needed
            groups=3         # Apply the same filter to each channel separately
        )
        
        # Set the weights to our mask values and make them non-learnable
        with torch.no_grad():
            # Reshape the mask for each input channel (RGB)
            mask_tensor = torch.tensor(self.mask, dtype=torch.float32)
            # Create a 4D tensor [out_channels, in_channels/groups, height, width]
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
            self.conv.weight = nn.Parameter(mask_tensor, requires_grad=False)
        
        # Move the model to the specified device
        self.conv = self.conv.to(device)
        
        print(f"Initialized RGB convolution with {self.kernel_size}x{self.kernel_size} mask")
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
        Apply the convolution to a batch of images.
        
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W]
            stride (int): Stride value for convolution
            
        Returns:
            torch.Tensor: Batch of convolved images [B, C, H, W]
        """
        # If stride is 1, we can use the pre-configured conv layer
        if stride == 1:
            with torch.no_grad():
                return self.conv(images)
        else:
            # For other strides, create a new temporary convolution
            with torch.no_grad():
                temp_conv = nn.Conv2d(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    padding=self.kernel_size // 2,
                    bias=False,
                    groups=3
                ).to(self.device)
                
                # Set the weights to our mask values
                temp_conv.weight = self.conv.weight
                
                # Apply convolution
                return temp_conv(images)

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
            transforms.ToTensor(),  # Convert PIL Image to tensor and scale to [0, 1]
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
                output_img = output_tensor[j].cpu().numpy().transpose(1, 2, 0)
                
                # Convert to uint8 range [0, 255]
                output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
                
                # Save the output image
                output_path = os.path.join(output_folder, img_file)
                Image.fromarray(output_img).save(output_path)
                
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Processed {total_images} images in {total_time:.4f} seconds " +
              f"({total_images/total_time:.2f} images/second)")
        
        return total_time


def main():
    parser = argparse.ArgumentParser(description='PyTorch 3D RGB Image Convolution')
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
    processor = RGBConvolution(args.mask_file, device=device)
    
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