import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

import os
import numpy as np
import itertools
from utils.metrics import FID
from utils.utils import save_images

Tensor = torch.cuda.FloatTensor
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

class AAE_CONV():
    def __init__(self,dataloader,params,exp_config,size):
        if torch.cuda.is_available():
            self.device='cuda'
        else:
            exit()
        self.sample_interval = exp_config['sample_interval']
        key = __file__.split('.')[0]
        self.models_path = f'{exp_config['models_saved']}/{key}/{size}x{size}'
        self.resume = exp_config['resume']
        self.images_path = f'{exp_config['images_saved']}/{key}/{size}x{size}'
        self.dataloader = dataloader
        self.mean = exp_config['mean']
        self.std = exp_config['std']
        os.makedirs(self.models_path,exist_ok=True)
        os.makedirs(self.images_path,exist_ok=True)
        
        self.img_shape = (params['channels'],size,size)
        self.latent_dim = params['latent_dim']
        self.lr = params['lr']
        self.betas = (params['b1'], params['b2'])
        
        #init models
        self.encoder = Encoder(self.img_shape).to(device=self.device)
        self.decoder = Decoder(self.img_shape,self.latent_dim).to(device=self.device)
        self.discriminator = Discriminator(self.latent_dim).to(device=self.device)
        self.params = params
        if self.resume:
            self.encoder.load_state_dict(torch.load(f'{self.models_path}/encoder.pth'))
            self.decoder.load_state_dict(torch.load(f'{self.models_path}/decoder.pth'))
            self.discriminator.load_state_dict(torch.load(f'{self.models_path}/discriminator.pth'))
        
        #optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.lr, betas=self.betas)  
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)
    
    def train(self,n_epochs):
        best_model = np.inf
        for epoch in range(n_epochs):
            best_fid = np.inf
            worst_fid=0
            for i, (imgs, _) in enumerate(self.dataloader):
                # Etiquetas para el discriminador (verdadero o falso)
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                encoded_imgs = self.encoder(real_imgs,self.latent_dim)
                decoded_imgs = self.decoder(encoded_imgs)

                # Loss measures generator's ability to fool the discriminator
                g_loss = 0.001 * adversarial_loss(self.discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
                    decoded_imgs, real_imgs
                )

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as discriminator ground truth
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(self.discriminator(z), valid)
                fake_loss = adversarial_loss(self.discriminator(encoded_imgs.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)

                d_loss.backward()
                self.optimizer_D.step()
                fid = FID(real_imgs,decoded_imgs)
                best_fid = fid if fid < best_fid else best_fid
                worst_fid = fid if fid > worst_fid else worst_fid
            if best_fid < best_model:
                torch.save(self.encoder.state_dict(),f'{self.models_path}/encoder.pth')
                torch.save(self.decoder.state_dict(),f'{self.models_path}/decoder.pth')
                torch.save(self.discriminator.state_dict(),f'{self.models_path}/discriminator.pth')
                
            print(
                "[Epoch %d/%d][D loss: %f] [G loss: %f] [B FID:%f] [W FID:%f]"
                % (epoch, n_epochs, d_loss.item(), g_loss.item(),best_fid,worst_fid)
            )
            if not epoch%self.sample_interval:
                save_images(decoded_imgs.data,os.path.join(self.images_path,f'{epoch}.jpg'),self.mean,self.std)
        save_images(decoded_imgs.data,os.path.join(self.images_path,f'{epoch}.jpg'),self.mean,self.std)

def reparamization(mu,logvar,latent_dim):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z

class Encoder(torch.nn.Module):
    def __init__(self,img_shape,latent_dim):
        in_channels,width,height=img_shape
        super(Encoder).__init__()
        self.conv_block=torch.nn.Sequential(
            nn.Conv2d(in_channels,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_size = self.get_conv_output_size(width)
        self.fc_block = nn.Sequential(
            nn.Linear(np.prod(self.final_size),1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mu(1024,latent_dim)
        self.log_var(1024,latent_dim)
        
    def get_conv_output_size(self, img_size):
        size = img_size
        channels = 3
        for layer in self.conv_block:
            if isinstance(layer, nn.Conv2d):
                # Compute output size for the current Conv2d layer with stride=1
                channels = layer.out_channels[0]
                size = (size + 2 * layer.padding[0] - layer.kernel_size[0])//layer.stride[0] + 1
        # The output size is the flattened vector size (channels * height * width)
        return [channels,size,size]  # 512 is the number of output channels from the last Conv2d
    
    def forward(self,img,latent_dim):
        #Pasar por el bloque convolucional
        img = self.conv_block(img)
        
        #Aplanar el tensor
        img_flat = img.view(img.shape[0], -1)
        
        img_flat = self.fc_block(img_flat)
        
        mu = self.mu(img_flat)
        log_var = self.log_var(img_flat)
        
        return reparamization(mu,log_var,latent_dim)
    
class Decoder(nn.Module):
    def __init__(self,latent_dim,in_channels,final_size):
        super(Decoder).__init__()
        self.final_size = final_size
        self.fc_block = nn.Sequential(
            nn.Linear(latent_dim,1024),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Linear(1024,np.prod(self.final_size)),
            nn.LeakyReLU(0.2,inplace=True)
        )
        
        self.conv_block=nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32,in_channels,kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )
        
    def forward(self,z):
        img_flat = self.fc_block(z)
        
        img = img_flat.view(img.size(0),*self.final_size)
        
        img = self.conv_block(img)
        
        return img
        
class Discriminator(nn.Module):
    def __init__(self,in_channels):
        super(Discriminator).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self,img):
        feats = self.model(img)
        flat_feats = feats.view(feats.size(0),-1)
        return torch.sigmoid(flat_feats)
        