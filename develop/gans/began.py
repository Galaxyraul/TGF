import argparse
import os
import numpy as np

from torchvision.utils import save_image

from utils.data_loader import DatasetLoader
from utils import metrics
from torch.autograd import Variable

import torch.nn as nn
import torch



parser = argparse.ArgumentParser()
parser.add_argument("--path",type=str,default='./dataset',help="Path of the dataset")
parser.add_argument("--load",type=bool,default=False,help='Load pretrained weights')
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="number of image channels")
opt = parser.parse_args()
print(opt)
filename = os.path.basename(__file__).split('.')[0]
base_path = f'{filename}/{opt.img_size}x{opt.img_size}'
models_path = f'models/{base_path}'
images_path = f'images/{base_path}'
os.makedirs(images_path, exist_ok=True)
os.makedirs(models_path,exist_ok=True)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

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
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(opt.channels, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = opt.img_size // 2
        down_dim = 64 * (opt.img_size // 2) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, opt.channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        out = self.fc(out.view(out.size(0), -1))
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

if opt.load:
    generator.load_state_dict(torch.load(f'{models_path}/generator.pth'))
    discriminator.load_state_dict(torch.load(f'{models_path}/discriminator.pth'))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

dl=DatasetLoader(opt.path,batch_size=opt.batch_size)
dataloader = dl.get_train() 

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

# BEGAN hyper parameters
gamma = 0.75
lambda_k = 0.001
k = 0.0
best_model = np.inf
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        best_fid = np.inf
        worst_fid=0 
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        d_real = discriminator(real_imgs)
        d_fake = discriminator(gen_imgs.detach())

        d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
        d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
        d_loss = d_loss_real - k * d_loss_fake

        d_loss.backward()
        optimizer_D.step()

        # ----------------
        # Update weights
        # ----------------

        diff = torch.mean(gamma * d_loss_real - d_loss_fake)

        # Update weight term for fake samples
        k = k + lambda_k * diff.item()
        k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

        # Update convergence metric
        M = (d_loss_real + torch.abs(diff))

        fid = metrics.FID(real_imgs,gen_imgs)
        best_fid = fid if fid < best_fid else best_fid
        worst_fid = fid if fid > worst_fid else worst_fid
        
    if best_fid < best_model:
        torch.save(generator.state_dict(),f'{models_path}/encoder.pth')
        torch.save(discriminator.state_dict(),f'{models_path}/discriminator.pth')
    print(
        "[Epoch %d/%d][D loss: %f] [G loss: %f] [B FID:%f] [W FID:%f]"
        % (epoch, opt.n_epochs, d_loss.item(), g_loss.item(),best_fid,worst_fid)
    )

    if not epoch % opt.sample_interval:
        save_image(gen_imgs.data[:25], f"{images_path}/{epoch}.png",nrow=5, normalize=True)
save_image(gen_imgs.data[:25], f"{images_path}/{epoch}.png",nrow=5, normalize=True)