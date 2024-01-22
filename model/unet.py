import torch, os
from pathlib import Path
from torch import nn, optim
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
import timm
from tqdm import tqdm
from torch.utils.data import DataLoader
from .tools import AverageMeter

# multi-GPU support  
from accelerate import Accelerator

def pretrain_generator(net_G:DynamicUnet, pretrain_dl:DataLoader, 
               epochs:int=20, lrate:float=1e-4, 
               stateDict:os.PathLike='runs\\pretrain', 
               resumeWeigths:os.PathLike='runs\\pretrain' ) -> DynamicUnet:
    
    stateDict  = Path(stateDict)
    accelerator = Accelerator(mixed_precision='bf16', project_dir=stateDict.parent)
    opt = optim.Adam(net_G.parameters(), lr=lrate)
    l1Loss = nn.L1Loss() 
    
    resumeWeigths = Path(resumeWeigths) if resumeWeigths is not None else None
    if resumeWeigths is not None and resumeWeigths.is_file():
        print(f"resuming from {resumeWeigths}")
        net_G.load_state_dict( torch.load(resumeWeigths) )

    pretrain_dl, net_G, opt = accelerator.prepare(
         pretrain_dl, net_G, opt
    )
    
    if resumeWeigths is not None and resumeWeigths.is_dir():
        print(f"resuming from {resumeWeigths}")
        accelerator.load_state(resumeWeigths)

    for e in range(epochs):
        loss_meter = AverageMeter()
        for pretrain_data in tqdm(pretrain_dl):
            L = pretrain_data['L'] 
            ab = pretrain_data['ab'] 
            preds = net_G(L)
            loss = l1Loss(preds, ab)
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()
            loss_meter.update(loss.item(), L.size(0))

        print(f"Epoch {e+1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.8f} after {loss_meter.count:.0f} items")
    
        accelerator.wait_for_everyone()
        accelerator.save_state(stateDict.parent)

    torch.save(net_G.state_dict(), stateDict )
    return net_G

def ResUnet(n_input:int=1, n_output:int=2, size:int=256, 
            timm_model_name:str='resnet18') -> DynamicUnet:
    model = timm.create_model(timm_model_name, pretrained=True)
    body =  create_body(model, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size))
    return net_G

class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x)

