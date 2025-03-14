import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Create directory to save images
os.makedirs("input_images", exist_ok=True)

# Load CIFAR-10 dataset
print("Downloading CIFAR-10 dataset...")
(train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()

# Save images to disk
num_images = 100  # Number of images to save
print(f"Saving {num_images} images to 'input_images' folder...")

for i in range(num_images):
    # Get the image
    img = train_images[i]
    
    # Save as PNG
    pil_img = Image.fromarray(img)
    filename = f"input_images/image_{i:03d}.png"
    pil_img.save(filename)

print(f"Successfully saved {num_images} images to 'input_images' folder")
print("Images are ready for testing your convolution implementation")