import torch
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

seed=42
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
if torch.are_deterministic_algorithms_enabled():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print("Deterministic algorithms are enabled")
else:
    print("Deterministic algorithms are not enabled")
    
path = './dataset_no_bc'
save_dir = './processed/data_no_bc_norm/300/drones'
i_t = os.path.join(save_dir,'train')
i_test = os.path.join(save_dir,'test')
i_v = os.path.join(save_dir,'val')

os.makedirs(i_t,exist_ok=True)
os.makedirs(i_test,exist_ok=True)
os.makedirs(i_v,exist_ok=True)


analysis_transforms = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=path,transform=analysis_transforms)
train_size = int(0.8*len(dataset))
val_size = int(0.1*len(dataset))
test_size = len(dataset) - train_size - val_size
train,test,val= random_split(dataset,[train_size,test_size,val_size])
print(train_size,test_size,val_size)
print(len(train),len(test),len(val))
data = DataLoader(train,shuffle=False,num_workers=4)
mean = 0
std = 0
total = 0
for images, _ in data:
    batch_count = images.size(0)
    images = images.view(batch_count,images.size(1),-1)
    mean += images.mean(2).sum(0)
    std += images.var(2).sum(0)
    total += batch_count
mean /= total
std =(std/total).sqrt()
print(f'Media:{mean}')
print(f'Desviación:{std}')

norm_transform=transforms.Compose([
    transforms.Normalize(mean=mean,std=std)
])

augment_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomAffine(degrees=30,translate=[0.2,0.2]),
])
n_augments = 5

for i,(img,_) in enumerate(train):
    norm_img = norm_transform(img)
    transforms.ToPILImage()(norm_img).save(os.path.join(i_t,f'{i}.jpg'))
    for n in range(n_augments):
        augmented = augment_transforms(norm_img)
        transforms.ToPILImage()(augmented).save(os.path.join(i_t,f'{i}_{n}.jpg'))

for i, (img, _) in enumerate(test):
    norm_img = norm_transform(img)
    transforms.ToPILImage()(norm_img).save(os.path.join(i_test,f'{i}.jpg'))
    
for i, (img, _) in enumerate(val):
    norm_img = norm_transform(img)
    transforms.ToPILImage()(norm_img).save(os.path.join(i_v,f'{i}.jpg'))
    




