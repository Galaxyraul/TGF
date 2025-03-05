import argparse
import os
import numpy as np
import itertools

from torchvision.utils import save_image
from utils.data_loader import DatasetLoader
from utils import metrics
from torch.autograd import Variable

import torch.nn as nn
import torch

def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor


class INFOGAN():
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
        os.makedirs(f"{self.images_path}/static", exist_ok=True)
        os.makedirs(f"{self.images_path}/varying_c1", exist_ok=True)
        os.makedirs(f"{self.images_path}/varying_c2", exist_ok=True)
        
        self.lambda_cat = params['lambda_cat']
        self.lambda_con = params['lambda_con']
        self.img_shape = (params['channels'],size,size)
        self.latent_dim = params['latent_dim']
        self.code_dim = params['code_dim']
        self.n_classes = params['n_classes']
        self.lr = params['lr']
        self.betas = (params['b1'], params['b2'])
        self.channels = params['channels']
        self.adversarial_loss = torch.nn.MSELoss().to(device=self.device)
        self.categorical_loss = torch.nn.CrossEntropyLoss().to(device=self.device)
        self.continuous_loss = torch.nn.MSELoss().to(device=self.device)
        self.Tensor = torch.cuda.FloatTensor
        self.static_z = Variable(FloatTensor(np.zeros((self.n_classes ** 2, self.latent_dim))))
        self.static_label = to_categorical(
            np.array([num for _ in range(self.n_classes) for num in range(self.n_classes)]), num_columns=self.n_classes
        )
        self.static_code = Variable(FloatTensor(np.zeros((self.n_classes ** 2, self.code_dim))))

        
        self.generator = Generator(self.latent_dim,size,self.channels,self.n_classes,self.code_dim).to(device=self.device)
        self.discriminator = Discriminator(self.channels,size,self.code_dim,self.n_classes).to(device=self.device)
        
        if self.resume:
            self.generator.load_state_dict(torch.load(f'{self.models_path}/generator.pth'))
            self.discriminator.load_state_dict(torch.load(f'{self.models_path}/discriminator.pth'))
        else:
            # Initialize weights
            self.generator.apply(weights_init_normal)
            self.discriminator.apply(weights_init_normal)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_info = torch.optim.Adam(
        itertools.chain(self.generator.parameters(), self.discriminator.parameters()), lr=self.lr, betas=self.betas)

        
    def sample_image(self,n_row, epoch):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Static sample
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.latent_dim))))
        static_sample = self.generator(z, self.static_label, self.static_code)
        save_image(static_sample.data, f"{self.images_path}/static/{epoch}.png", nrow=n_row, normalize=True)

        # Get varied c1 and c2
        zeros = np.zeros((n_row ** 2, 1))
        c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
        c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
        c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))
        sample1 = self.generator(self.static_z, self.static_label, c1)
        sample2 = self.generator(self.static_z, self.static_label, c2)
        save_image(sample1.data, f"{self.images_path}/varying_c1/{epoch}.png" , nrow=n_row, normalize=True)
        save_image(sample2.data, f"{self.images_path}/varying_c2/{epoch}.png", nrow=n_row, normalize=True)

    def train(self,n_epochs):
        best_model = np.inf
        for epoch in range(n_epochs):
            best_fid = np.inf
            worst_fid=0
            for i, (imgs, labels) in enumerate(self.dataloader):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(FloatTensor))
                labels = to_categorical(labels.numpy(), num_columns=self.n_classes)

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
                label_input = to_categorical(np.random.randint(0, self.n_classes, batch_size), num_columns=self.n_classes)
                code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, self.code_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z, label_input, code_input)

                # Loss measures generator's ability to fool the discriminator
                validity, _, _ = self.discriminator(gen_imgs)
                g_loss = self.adversarial_loss(validity, valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Loss for real images
                real_pred, _, _ = self.discriminator(real_imgs)
                d_real_loss = self.adversarial_loss(real_pred, valid)

                # Loss for fake images
                fake_pred, _, _ = self.discriminator(gen_imgs.detach())
                d_fake_loss = self.adversarial_loss(fake_pred, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                # ------------------
                # Information Loss
                # ------------------

                self.optimizer_info.zero_grad()

                # Sample labels
                sampled_labels = np.random.randint(0, self.n_classes, batch_size)

                # Ground truth labels
                gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

                # Sample noise, labels and code as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
                label_input = to_categorical(sampled_labels, num_columns=self.n_classes)
                code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, self.code_dim))))

                gen_imgs = self.generator(z, label_input, code_input)
                _, pred_label, pred_code = self.discriminator(gen_imgs)

                info_loss = self.lambda_cat * self.categorical_loss(pred_label, gt_labels) + self.lambda_con * self.continuous_loss(
                    pred_code, code_input
                )

                info_loss.backward()
                self.optimizer_info.step()

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

            if not epoch%self.sample_interval:
                self.sample_image(n_row=10, epoch=epoch)
        self.sample_image(n_row=10, epoch=epoch)
            
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self,latent_dim,img_size,channels,n_classes,code_dim):
        super(Generator, self).__init__()
        input_dim = latent_dim + n_classes + code_dim

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
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

    def forward(self, noise, labels, code):
        print(noise.shape)
        print(labels.shape)
        print(code.shape)
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self,channels,img_size,code_dim,n_classes):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code

