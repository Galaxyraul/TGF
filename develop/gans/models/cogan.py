import os
import numpy as np

from torchvision.utils import save_image
from utils import metrics
from torch.autograd import Variable


import torch.nn as nn
import torch

class COGAN():
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
        self.img_size = size
        self.adversarial_loss = torch.nn.MSELoss().to(device=self.device)
        self.Tensor = torch.cuda.FloatTensor
        
        self.generator = CoupledGenerators(self.latent_dim,size,params['channels']).to(device=self.device)
        self.discriminator = CoupledDiscriminators(params['channels'],size).to(device=self.device)
        if self.resume:
            self.generator.load_state_dict(torch.load(f'{self.models_path}/generator.pth'))
            self.discriminator.load_state_dict(torch.load(f'{self.models_path}/discriminator.pth'))
        else:
            self.generator.apply(weights_init_normal)
            self.discriminator.apply(weights_init_normal)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        
    def train(self,n_epochs):
        best_model = np.inf
        for epoch in range(n_epochs):
            for i, ((imgs1, _), (imgs2, _)) in enumerate(zip(self.dataloader, self.dataloader)):
                best_fid = np.inf
                worst_fid=0 
                batch_size = imgs1.shape[0]

                # Adversarial ground truths
                valid = Variable(self.Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                imgs1 = Variable(imgs1.type(self.Tensor).expand(imgs1.size(0), 3, self.img_size, self.img_size))
                imgs2 = Variable(imgs2.type(self.Tensor))

                # ------------------
                #  Train Generators
                # ------------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))

                # Generate a batch of images
                gen_imgs1, gen_imgs2 = self.generator(z)
                # Determine validity of generated images
                validity1, validity2 = self.discriminator(gen_imgs1, gen_imgs2)

                g_loss = (self.adversarial_loss(validity1, valid) + self.adversarial_loss(validity2, valid)) / 2

                g_loss.backward()
                self.optimizer_G.step()

                # ----------------------
                #  Train Discriminators
                # ----------------------

                self.optimizer_D.zero_grad()

                # Determine validity of real and generated images
                validity1_real, validity2_real = self.discriminator(imgs1, imgs2)
                validity1_fake, validity2_fake = self.discriminator(gen_imgs1.detach(), gen_imgs2.detach())

                d_loss = (
                    self.adversarial_loss(validity1_real, valid)
                    + self.adversarial_loss(validity1_fake, fake)
                    + self.adversarial_loss(validity2_real, valid)
                    + self.adversarial_loss(validity2_fake, fake)
                ) / 4

                d_loss.backward()
                self.optimizer_D.step()
                
                fid = metrics.FID(imgs1,gen_imgs1)
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
                gen_imgs = torch.cat((gen_imgs1.data, gen_imgs2.data), 0)
                save_image(gen_imgs, f"{self.images_path}/{epoch}.png", nrow=8, normalize=True)
        save_image(gen_imgs, f"{self.images_path}/{epoch}.png", nrow=8, normalize=True)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class CoupledGenerators(nn.Module):
    def __init__(self,latent_dim,img_size,channels):
        super(CoupledGenerators, self).__init__()

        self.init_size = img_size // 4
        self.fc = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.shared_conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
        )
        self.G1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.G2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.fc(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img_emb = self.shared_conv(out)
        img1 = self.G1(img_emb)
        img2 = self.G2(img_emb)
        return img1, img2


class CoupledDiscriminators(nn.Module):
    def __init__(self,channels,img_size):
        super(CoupledDiscriminators, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block

        self.shared_conv = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.D1 = nn.Linear(128 * ds_size ** 2, 1)
        self.D2 = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img1, img2):
        # Determine validity of first image
        out = self.shared_conv(img1)
        out = out.view(out.shape[0], -1)
        validity1 = self.D1(out)
        # Determine validity of second image
        out = self.shared_conv(img2)
        out = out.view(out.shape[0], -1)
        validity2 = self.D2(out)

        return validity1, validity2

