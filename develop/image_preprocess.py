import torch
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms
import os
from PIL import Image
seed=42
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
if torch.are_deterministic_algorithms_enabled():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print("Deterministic algorithms are enabled")
else:
    print("Deterministic algorithms are not enabled")
            
path = './dataset'
directory = os.listdir(path)
n_augmentations = 5
#Preprocessing
sizes = [300]
def return_output_dir(size):
    return f'./datasets_augmented/{size}x{size}'

for size in sizes:
    os.makedirs(f'{return_output_dir(size)}/train',exist_ok=True)
    os.makedirs(f'{return_output_dir(size)}/test',exist_ok=True)
    os.makedirs(f'{return_output_dir(size)}/val',exist_ok=True)

def size_transform(size):
    return transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor()
    ])

mean=[0.5789, 0.6016, 0.6047]
std=[0.2234, 0.2208, 0.2317]
transform_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomAffine(degrees=30,translate=[0.2,0.2]),
    transforms.Normalize(mean=mean,std=std)
])

dataset = datasets.ImageFolder(root=path)
train_size = int(0.8*len(dataset))
val_size = int(0.1*len(dataset))
test_size = len(dataset) - train_size - val_size
train,test,val= random_split(dataset,[train_size,test_size,val_size])

for i,(img,label) in enumerate(train):
    for size in sizes:
        size_transformation = size_transform(size)
        resized_img = size_transformation(img)
        transforms.ToPILImage()(resized_img).save(os.path.join(f'{return_output_dir(size)}/train',f'{i}.jpg'))
    for j in range(n_augmentations):
        augmented_image=transform_augment(resized_img)
        for size in sizes:
            size_transformation = size_transform(size)
            resized_img = size_transformation(augmented_image)
            transforms.ToPILImage()(resized_img).save(os.path.join(f'{return_output_dir(size)}/train',f'{i}_{j}.jpg'))

for i,(img,label) in enumerate(test):
    for size in sizes:
        size_transformation = size_transform(size)
        resized_img = size_transformation(img)
        transforms.ToPILImage()(resized_img).save(os.path.join(f'{return_output_dir(size)}/test',f'{i}.jpg'))
    
for i,(img,label) in enumerate(val):
    for size in sizes:
        size_transformation = size_transform(size)
        resized_img = size_transformation(img)
        transforms.ToPILImage()(resized_img).save(os.path.join(f'{return_output_dir(size)}/val',f'{i}.jpg'))