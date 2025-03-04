# Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
import os
import numpy as np

from torchvision.utils import save_image
from utils import metrics
from torch.autograd import Variable

import torch.nn as nn
import torch


class BGAN():
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
        self.discriminator_loss = torch.nn.BCELoss().to(device=self.device)
        self.Tensor = torch.cuda.FloatTensor
        
        self.generator = Generator(self.latent_dim,self.img_shape).to(device=self.device)
        self.discriminator = Discriminator(self.img_shape).to(device=self.device)
        if self.resume:
            self.generator.load_state_dict(torch.load(f'{self.models_path}/generator.pth'))
            self.discriminator.load_state_dict(torch.load(f'{self.models_path}/discriminator.pth'))
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        
    def train(self,n_epochs):
        best_model = np.inf
        for epoch in range(n_epochs):
            for i, (imgs, _) in enumerate(self.dataloader):
                best_fid = np.inf
                worst_fid=0 
                # Adversarial ground truths
                valid = Variable(self.Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z,self.img_shape)

                # Loss measures generator's ability to fool the discriminator
                g_loss = boundary_seeking_loss(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.discriminator_loss(self.discriminator(real_imgs), valid)
                fake_loss = self.discriminator_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()
            
                fid = metrics.FID(real_imgs,gen_imgs)
                best_fid = fid if fid < best_fid else best_fid
                worst_fid = fid if fid > worst_fid else worst_fid
            if best_fid < best_model:
                torch.save(self.generator.state_dict(),f'{self.models_path}/encoder.pth')
                torch.save(self.discriminator.state_dict(),f'{self.models_path}/discriminator.pth')
            print(
                "[Epoch %d/%d][D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, d_loss.item(), g_loss.item())
            )

            if not epoch % self.sample_interval:
                save_image(gen_imgs.data[:25], f"./{self.images_path}/{epoch}.png",nrow=5, normalize=True)
        save_image(gen_imgs.data[:25], f"./{self.images_path}/{epoch}.png",nrow=5, normalize=True)
        
        
        
class Generator(nn.Module):
    def __init__(self,latent_dim,img_shape):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, img_shape):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self,img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


def boundary_seeking_loss(y_pred, y_true):
    """
    Boundary seeking loss.
    Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
    """
    return 0.5 * torch.mean((torch.log(y_pred) - torch.log(1 - y_pred)) ** 2)


# Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
import os
import numpy as np

from torchvision.utils import save_image
from utils import metrics
from torch.autograd import Variable

import torch.nn as nn
import torch


class BGAN():
    def __init__(self,dataloader,params,size,sample_interval,models_path,resume,images_path):
        if torch.cuda.is_available():
            self.device='cuda'
        else:
            exit()
        self.dataloader = dataloader
        self.images_path = images_path
        self.models_path = models_path
        self.sample_interval = sample_interval
        os.makedirs(self.models_path,exist_ok=True)
        os.makedirs(self.images_path,exist_ok=True)
        
        self.img_shape = (params['channels'],size,size)
        self.latent_dim = params['latent_dim']
        self.lr = params['lr']
        self.betas = (params['b1'], params['b2'])
        self.discriminator_loss = torch.nn.BCELoss().to(device=self.device)
        self.Tensor = torch.cuda.FloatTensor
        
        self.generator = Generator(self.latent_dim,self.img_shape).to(device=self.device)
        self.discriminator = Discriminator(self.img_shape).to(device=self.device)
        if resume:
            self.generator.load_state_dict(torch.load(f'{self.models_path}/generator.pth'))
            self.discriminator.load_state_dict(torch.load(f'{self.models_path}/discriminator.pth'))
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        
    def train(self,n_epochs):
        best_model = np.inf
        for epoch in range(n_epochs):
            for i, (imgs, _) in enumerate(self.dataloader):
                best_fid = np.inf
                worst_fid=0 
                # Adversarial ground truths
                valid = Variable(self.Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z,self.img_shape)

                # Loss measures generator's ability to fool the discriminator
                g_loss = boundary_seeking_loss(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.discriminator_loss(self.discriminator(real_imgs), valid)
                fake_loss = self.discriminator_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()
            
                fid = metrics.FID(real_imgs,gen_imgs)
                best_fid = fid if fid < best_fid else best_fid
                worst_fid = fid if fid > worst_fid else worst_fid
            if best_fid < best_model:
                torch.save(self.generator.state_dict(),f'{self.models_path}/encoder.pth')
                torch.save(self.discriminator.state_dict(),f'{self.models_path}/discriminator.pth')
            print(
                "[Epoch %d/%d][D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, d_loss.item(), g_loss.item())
            )

            if not epoch % self.sample_interval:
                save_image(gen_imgs.data[:25], f"./{self.images_path}/{epoch}.png",nrow=5, normalize=True)
        save_image(gen_imgs.data[:25], f"./{self.images_path}/{epoch}.png",nrow=5, normalize=True)
        
        
        
class Generator(nn.Module):
    def __init__(self,latent_dim,img_shape):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, img_shape):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self,img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


def boundary_seeking_loss(y_pred, y_true):
    """
    Boundary seeking loss.
    Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
    """
    return 0.5 * torch.mean((torch.log(y_pred) - torch.log(1 - y_pred)) ** 2)








