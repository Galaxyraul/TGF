from .aae import *
from .acgan import *
from .began import *
from .bgan import *
from .cgan import *
from .cogan import *
from .dcgan import *
from .dragan import *
from .ebgan import *
from .gan import *
from .infogan import *
from .lsgan import *
from .relativistic_gan import *
from .sgan import *
from .wgan_div import *
from .wgan_gp import *
from .wgan import *

models = {
    'aae':AAE,
    'acgan':ACGAN,
    'began':BEGAN,
    'bgan':BGAN,
    'cgan':CGAN,
    'cogan':COGAN,
    'dcgan':DCGAN,
    'dragan':DRAGAN,
    'ebgan':EBGAN,
    'gan':GAN,
    #'infogan':INFOGAN, #REVISAR
    'lsgan':LSGAN,
    'relativistic_gan':RELATIVISTIC_GAN,
    'sgan':SGAN,
    'wgan_div':WGAN_DIV,
    'wgan_gp':WGAN_GP,
    'wgan':WGAN
    
}