import numpy as np
import os
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf

# Create directory to save images
os.makedirs("input_images", exist_ok=True)

# Option 1: Use TensorFlow datasets - Flowers dataset (larger than CIFAR)
print("Downloading Flowers dataset...")
try:
    # Load the flowers dataset which has larger images (various sizes, typically ~500x500)
    flowers_dataset = tf.keras.utils.get_file(
        'flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar=True)
    
    # Get image paths
    image_paths = []
    classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    for flower_class in classes:
        class_dir = os.path.join(flowers_dataset, flower_class)
        if os.path.isdir(class_dir):
            files = os.listdir(class_dir)
            image_paths.extend([os.path.join(class_dir, file) for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Limit to 100 images
    image_paths = image_paths[:100]
    
    print(f"Copying {len(image_paths)} flower images...")
    for i, img_path in enumerate(image_paths):
        try:
            # Open and resize the image to a consistent size (optional)
            img = Image.open(img_path)
            
            # Optional: resize to a consistent size if desired
            # img = img.resize((256, 256))
            
            # Save as PNG
            filename = f"input_images/flower_{i:03d}.png"
            img.save(filename)
            print(f"Saved {filename} ({img.width}x{img.height})")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Successfully saved flower images to 'input_images' folder")

except Exception as e:
    print(f"Error with flower dataset: {e}")
    
    # Option 2: As fallback, download a few images from an alternative source
    print("Falling back to downloading from Microsoft COCO dataset thumbnails...")
    
    # Some sample COCO image URLs (resized thumbnails)
    image_urls = [
        "https://farm9.staticflickr.com/8072/8346734966_f9cd7d3796_z.jpg",
        "https://farm2.staticflickr.com/1238/1077992716_40e485cd7c_z.jpg",
        "https://farm4.staticflickr.com/3586/3428771552_44865f955f_z.jpg",
        "https://farm6.staticflickr.com/5142/5603140646_57f3b0c9a7_z.jpg",
        "https://farm9.staticflickr.com/8366/8452262407_aabafe8cae_z.jpg"
    ]
    
    # Download and save images
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            filename = f"input_images/coco_{i:03d}.jpg"
            img.save(filename)
            print(f"Saved {filename} ({img.width}x{img.height})")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

print("Images are ready for testing your convolution implementation")