import torch, os
from skimage.color import lab2rgb
from skimage.util import random_noise
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0., 0., 0.]
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).to(torch.float32).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def visualize(model, data, epoch, save_dir=''):
    with torch.inference_mode():
        model.setup_input(data)
        model.forward()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f'Colorisation at Epoch {epoch+1}')
    for i in range(4):
        ax = plt.subplot(3, 4, i + 1)
        ax.imshow(L[i][0].to(torch.float32).cpu(), cmap='gray')
        ax.set_title("INPUT GRAY")
        ax.axis("off")
        ax = plt.subplot(3, 4, i + 5)
        ax.imshow(fake_imgs[i])
        ax.set_title("FAKE")
        ax.axis("off")
        ax = plt.subplot(3, 4, i + 9)
        ax.imshow(real_imgs[i])
        ax.set_title("REAL")
        ax.axis("off")

    if Path(save_dir).exists():
        fig.savefig( Path(save_dir) / f"colorization_epoch_{epoch+1}.png")
    else:
        plt.show()
        
def log_results(loss_meter_dict:dict, logFile:os.PathLike):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")
    
    with open(logFile , 'a') as log:
        log.write( ";".join( str(i.avg) for i in loss_meter_dict.values()) + "\n")


def grainify(img:np.ndarray):
    c, rows, cols = img.shape
    val = np.random.uniform(0.036, 0.107)**2 

    # Full resolution
    noise_1 = np.zeros((rows, cols))
    noise_1 = random_noise(noise_1, mode='gaussian', var=val, clip=False)

    # # Half resolution
    noise_2 = np.zeros((rows//2, cols//2))
    noise_2 = random_noise(noise_2, mode='gaussian', var=(val*2)**2, clip=False)  
    noise_2 = resize(noise_2, (rows, cols))  # Upscale to original image size

    noise = noise_1 + noise_2 
    noise = np.stack( [noise]*c, axis=0)
    
    noisy_img = img/255 + noise # Add noise_im to the input image.
    return np.round((255 * noisy_img)).clip(0, 255).astype(np.uint8)