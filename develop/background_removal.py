import rembg
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
from tqdm import tqdm
import onnxruntime as ort  # GPU inference backend

# Paths
images_path = './dataset/drones'
save_path = './dataset_no_bc_2/drones'
os.makedirs(save_path, exist_ok=True)

# Enable GPU processing using ONNX Runtime
session = rembg.new_session("u2net", providers=["CUDAExecutionProvider"])  # Use GPU

for image in tqdm(os.listdir(images_path)):
    img_path = os.path.join(images_path, image)
    
    # Load image and ensure RGBA (to keep transparency)
    input_image = Image.open(img_path).convert("RGBA")

    # Enhance contrast slightly to improve background removal
    enhancer = ImageEnhance.Contrast(input_image)
    input_image = enhancer.enhance(1.2)  # Slight boost to avoid color loss

    # Convert to NumPy array
    input_array = np.array(input_image)

    # Remove background using GPU-accelerated rembg
    output_array = rembg.remove(input_array, session=session)

    # Convert back to an image
    output_image = Image.fromarray(output_array)

    # Post-processing to refine the result
    output_image = output_image.filter(ImageFilter.SMOOTH)  # Smooth edges
    output_image = output_image.crop(output_image.getbbox())  # Auto-crop transparent areas

    # Save with original colors and transparency
    save_name = os.path.join(save_path, image.split('.')[0] + '.png')
    output_image.save(save_name, format="PNG")

print("✅ GPU-accelerated background removal completed!")
