import torch
from torchvision import models
from torchvision.transforms import functional as F
import numpy as np
from scipy.linalg import sqrtm
inception = models.inception_v3(pretrained=True)
inception.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception.to(device)

def transform_tensors(tensors):
    t_tensors=[]
    for tensor in tensors:
        t_tensor = F.resize(tensor,299)
        t_tensor = F.center_crop(tensor,299)
        t_tensors.append(t_tensor)
    return torch.stack(t_tensors)

def extract_features(image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        features = inception(image_tensor)
    return features.cpu().numpy()

def FID(real,generated):
    real = transform_tensors(real)
    generated = transform_tensors(generated)
    f_real = extract_features(real)
    f_gen = extract_features(generated)
    
    mu_real = np.mean(f_real)
    mu_gen = np.mean(f_gen)
    cov_real = np.cov(f_real)
    cov_gen = np.cov(f_gen)
    
    diff = mu_real - mu_gen
    cov_sqrt = sqrtm(cov_real.dot(cov_gen))
    fid = np.sum(diff**2) + np.trace(cov_real + cov_gen - 2 * cov_sqrt)
    return fid
