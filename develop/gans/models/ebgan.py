import argparse
import os
import numpy as np

from utils.data_loader import DatasetLoader
from utils import metrics
from torchvision.utils import save_image

from torch.autograd import Variable

import torch.nn as nn
import torch

class EBGAN():
    def __init__(self,dataloader,params,exp_config,size):
        if torch.cuda.is_available():
            self.device='cuda'
        else:
            exit()
        self.sample_interval = exp_config['sample_interval']
        key = os.path.basename(__file__).split('.')[0]
        self.models_path = f'{exp_config['models_saved']}/{key}/{size}x{size}'
        self.resume = exp_config['resume']
        self.images_path = f'{exp_config['images_saved']}/{key}/{size}x{size}'
        self.dataloader = dataloader
        os.makedirs(self.models_path,exist_ok=True)
        os.makedirs(self.images_path,exist_ok=True)
    
        self.batch_size = exp_config['batch_size']
        self.margin = max(1, self.batch_size / 64.0)
        channels = params['channels']
        self.img_shape = (channels,size,size)
        self.latent_dim = params['latent_dim']
        self.lr = params['lr']
        self.betas = (params['b1'], params['b2'])
        self.pixelwise_loss = nn.MSELoss().to(device=self.device)
        self.Tensor = torch.cuda.FloatTensor
        
        self.generator = Generator(self.latent_dim,size,channels).to(device=self.device)
        self.discriminator = Discriminator(channels,size).to(device=self.device)
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
                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z)
                recon_imgs, img_embeddings = self.discriminator(gen_imgs)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.pixelwise_loss(recon_imgs, gen_imgs.detach()) + self.lambda_pt * pullaway_loss(img_embeddings)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_recon, _ = self.discriminator(real_imgs)
                fake_recon, _ = self.discriminator(gen_imgs.detach())

                d_loss_real = self.pixelwise_loss(real_recon, real_imgs)
                d_loss_fake = self.pixelwise_loss(fake_recon, gen_imgs.detach())

                d_loss = d_loss_real
                if (self.margin - d_loss_fake.data).item() > 0:
                    d_loss += self.margin - d_loss_fake

                d_loss.backward()
                self.optimizer_D.step()

                fid = metrics.FID(real_imgs,gen_imgs)
                best_fid = fid if fid < best_fid else best_fid
                worst_fid = fid if fid > worst_fid else worst_fid
                
            if best_fid < best_model:
                torch.save(self.generator.state_dict(),f'{self.models_path}/encoder.pth')
                torch.save(self.discriminator.state_dict(),f'{self.models_path}/discriminator.pth')
            print(
                "[Epoch %d/%d][D loss: %f] [G loss: %f] [B FID:%f] [W FID:%f]"
                % (epoch, n_epochs, d_loss.item(), g_loss.item(),best_fid,worst_fid)
            )

            if not epoch % self.sample_interval:
                save_image(gen_imgs.data[:25], f"{self.images_path}/{epoch}.png",nrow=5, normalize=True)
        save_image(gen_imgs.data[:25], f"{self.images_path}/{epoch}.png",nrow=5, normalize=True)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self,latent_dim,img_size,channels):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self,channels,img_size):
        super(Discriminator, self).__init__()

        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(channels, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = img_size // 2
        down_dim = 64 * (img_size // 2) ** 2

        self.embedding = nn.Linear(down_dim, 32)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        embedding = self.embedding(out.view(out.size(0), -1))
        out = self.fc(embedding)
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out, embedding

def pullaway_loss(embeddings):
    norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True))
    normalized_emb = embeddings / norm
    similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
    batch_size = embeddings.size(0)
    loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return loss_pt

