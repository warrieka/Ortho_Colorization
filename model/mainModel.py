import torch
from torch import nn, optim
from accelerate import Accelerator
from .discriminator import PatchDiscriminator
from .unet import Unet

# GAN Loss
class GANLoss(nn.Module):
    def __init__(self, accelerator:Accelerator=None):
        super().__init__()
        if accelerator and accelerator.mixed_precision == 'bf16':
            self.dtype = torch.bfloat16 
        elif accelerator and accelerator.mixed_precision == 'fp16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.device = accelerator.device if accelerator else torch.device(
                                            "cuda" if torch.cuda.is_available() else "cpu") 
        self.register_buffer('real_label', torch.tensor(True, dtype=self.dtype ))
        self.register_buffer('fake_label', torch.tensor(False, dtype=self.dtype ))
        if accelerator:
           self.loss = accelerator.prepare( nn.BCEWithLogitsLoss() )
        else:
           self.loss = nn.BCEWithLogitsLoss()
    
    def __call__(self, preds, target_is_real):
        labels = torch.tensor(target_is_real, dtype=self.dtype).expand_as(preds)
        loss = self.loss(preds, labels.to(self.device) )
        return loss


# Compose Main Model
class MainModel(nn.Module):
    def __init__(self, net_G:nn.Module=None, net_D:nn.Module=None, lr_G:float=2e-4, lr_D:float=2e-4, 
                 beta1:float=0.5, beta2:float=0.999, lambda_L1:float=100., accelerator:Accelerator=None):
        super().__init__()
        
        self.accelerator = accelerator
        self.device = accelerator.device if accelerator else torch.device(
                                            "cuda" if torch.cuda.is_available() else "cpu") 
        self.lambda_L1 = lambda_L1

        if net_G is None:
            self.net_G = self.init_weights(Unet(input_c=1, output_c=2, n_down=8, num_filters=64)).to(self.device)
        else:
            self.net_G = net_G.to(self.device)

        if net_D is None:
            self.net_D = self.init_weights(PatchDiscriminator(input_c=3, n_down=3, num_filters=64)).to(self.device)
        else:
            self.net_D = net_D
        
        self.GAN_loss = GANLoss(self.accelerator).to(self.device)
        self.L1_loss =  nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

        if self.accelerator:
            self.net_G, self.net_D, self.GAN_loss, self.opt_G, self.opt_D = self.accelerator.prepare( 
                                     self.net_G, self.net_D, self.GAN_loss, self.opt_G, self.opt_D )
    @staticmethod
    def init_weights(net, init:str='norm', gain:float=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and 'Conv' in classname:
                if init == 'norm':
                    nn.init.normal_(m.weight.data, mean=0.0, std=gain)
                elif init == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif 'BatchNorm2d' in classname:
                nn.init.normal_(m.weight.data, 1., gain)
                nn.init.constant_(m.bias.data, 0.)

        net.apply(init_func)
        return net

    def set_requires_grad(self, model:nn.Module, requires_grad:bool=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data:dict):
        self.L =  data['L'].to(self.device)
        self.ab = data['ab'].to(self.device) if 'ab' in data else None
        
    def forward(self):
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GAN_loss(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GAN_loss(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        if self.accelerator:
            self.accelerator.backward(self.loss_D)
        else:
            self.loss_D.backward()


    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GAN_loss(fake_preds, True)
        self.loss_G_L1 = self.L1_loss(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        if self.accelerator:
            self.accelerator.backward(self.loss_G)
        else:
            self.loss_G.backward()
    
    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
