import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class DatasetLoader:
    def __init__(self,path,batch_size,n_cpu):
        
        self.train_ds = datasets.ImageFolder(root=os.path.join(path,'train')) 
        self.test_ds = datasets.ImageFolder(root=os.path.join(path,'test')) 
        self.val_ds = datasets.ImageFolder(root=os.path.join(path,'val')) 
        
        self.train = DataLoader(self.train_ds,batch_size=batch_size,shuffle=True,num_workers=n_cpu)
        self.test = DataLoader(self.train_ds,batch_size=batch_size,shuffle=False,num_workers=n_cpu)
        self.val = DataLoader(self.train_ds,batch_size=batch_size,shuffle=False,num_workers=n_cpu)
    
    def get_train(self):
        return self.train
    def get_test(self):
        return self.test
    def get_val(self):
        return self.val
