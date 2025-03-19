import os
import numpy as np
import itertools

from torchvision.utils import save_image
from utils import metrics
from torch.autograd import Variable
import torch.nn as nn
import torch

class AAE():
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
        os.makedirs(self.models_path,exist_ok=True)
        os.makedirs(self.images_path,exist_ok=True)
        
        self.img_shape = (params['channels'],size,size)
        self.latent_dim = params['latent_dim']
        self.lr = params['lr']
        self.betas = (params['b1'], params['b2'])
        self.adversarial_loss = torch.nn.BCELoss()
        self.pixelwise_loss = torch.nn.L1Loss()
        self.Tensor = torch.cuda.FloatTensor
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
                valid = Variable(self.Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                encoded_imgs = self.encoder(real_imgs,self.reparameterization)
                decoded_imgs = self.decoder(encoded_imgs,self.img_shape)

                # Loss measures generator's ability to fool the discriminator
                g_loss = 0.001 * self.adversarial_loss(self.discriminator(encoded_imgs), valid) + 0.999 * self.pixelwise_loss(
                    decoded_imgs, real_imgs
                )

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as discriminator ground truth
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))
                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(z), valid)
                fake_loss = self.adversarial_loss(self.discriminator(encoded_imgs.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)

                d_loss.backward()
                self.optimizer_D.step()
                fid = metrics.FID(real_imgs,decoded_imgs)
                best_fid = fid if fid < best_fid else best_fid
                worst_fid = fid if fid > worst_fid else worst_fid
        #TODO
            if best_fid < best_model:
                torch.save(self.encoder.state_dict(),f'{self.models_path}/encoder.pth')
                torch.save(self.decoder.state_dict(),f'{self.models_path}/decoder.pth')
                torch.save(self.discriminator.state_dict(),f'{self.models_path}/discriminator.pth')
                
            print(
                "[Epoch %d/%d][D loss: %f] [G loss: %f] [B FID:%f] [W FID:%f]"
                % (epoch, n_epochs, d_loss.item(), g_loss.item(),best_fid,worst_fid)
            )
            if not epoch%self.sample_interval:
                self.sample_image(n_row=10, batches_done=epoch)
        self.sample_image(n_row=10, batches_done=epoch)
        
    def reparameterization(self,mu,logvar):
        std = torch.exp(logvar / 2)
        sampled_z = Variable(self.Tensor(np.random.normal(0, 1, (mu.size(0), self.latent_dim))))
        z = sampled_z * std + mu
        return z
    
    def sample_image(self,n_row, batches_done):
        """Saves a grid of generated digits"""
        # Sample noise
        z = Variable(self.Tensor(np.random.normal(0, 1, (n_row ** 2, self.latent_dim))))
        gen_imgs = self.decoder(z,self.img_shape)
        save_image(gen_imgs.data, f"{self.images_path}/{batches_done}.png", nrow=n_row, normalize=True)


class Encoder(nn.Module):
    def __init__(self,img_shape,latent_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512,latent_dim)
        self.logvar = nn.Linear(512,latent_dim)

    def forward(self, img,reparameterization):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self,img_shape,latent_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z,img_shape):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self,latent_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

