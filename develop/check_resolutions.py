from PIL import Image
import os

# Path to your folder containing the images
image_folder = "Dataset"
resol_dict = {}
# Loop through all the files in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # or other image formats
        img_path = os.path.join(image_folder, filename)
        
        # Open the image and get its resolution
        with Image.open(img_path) as img:
            width, height = img.size
            try:
                resol_dict[f'{width}x{height}']+=1
            except:
                resol_dict[f'{width}x{height}']=1
print(resol_dict)
