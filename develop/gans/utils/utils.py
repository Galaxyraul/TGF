from torchvision.utils import save_image
from torchvision import transforms

def save_images(imgs,path,mean,std):
    mean = [-x for x in mean]
    std = [1/x for x in std]
    undo_norm = transforms.Normalize(mean=mean,std=std)
    imgs = [undo_norm(img) for img in imgs]
    save_image(imgs,path,n_row=10,normalize=False)