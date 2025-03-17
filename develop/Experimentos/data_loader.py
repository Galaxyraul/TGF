import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class DatasetLoader:
    def __init__(self,path,batch_size,norm):
        transformations = [transforms.ToTensor()]
        if norm:
            transformations.append(transforms.Normalize(mean=[0.5789, 0.6016, 0.6047],std=[0.2234, 0.2207, 0.2316]))
        transform=transforms.Compose(transformations)
        
        self.train_ds = datasets.ImageFolder(root=os.path.join(path,'train'),transform=transform) 
        self.test_ds = datasets.ImageFolder(root=os.path.join(path,'test'),transform=transform) 
        self.val_ds = datasets.ImageFolder(root=os.path.join(path,'val'),transform=transform) 
        
        self.train = DataLoader(self.train_ds,batch_size=batch_size,shuffle=True)
        self.test = DataLoader(self.train_ds,batch_size=batch_size,shuffle=False)
        self.val = DataLoader(self.train_ds,batch_size=batch_size,shuffle=False)
    
    def get_train(self):
        return self.train
    def get_test(self):
        return self.test
    def get_val(self):
        return self.val
