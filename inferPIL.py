import numpy as np
from PIL import Image
import torch, shutil
from pathlib import Path
from model.mainModel import MainModel
from accelerate import Accelerator
from model.tools import lab_to_rgb

IMSIZE = 512
MODEL =  r".\runs\model_pil_run10_1m_512.pth"
IN_DIR = r"W:\testdata\tiles_1995_gray"
OUT_DIR = r"W:\testdata\tiles_1995_rgb"
FORMAT = 'png' #or 'jpg'

if __name__ == '__main__':
    accelerator = Accelerator(mixed_precision='fp16')
    model = MainModel(accelerator=accelerator)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load( MODEL , map_location=device ) )
    paths = Path( IN_DIR ).glob(f'*.{FORMAT}')

    for path in paths:
        wld = path.parent / f"{path.stem}.wld"
        out_png = Path(OUT_DIR) / path.name
        img = Image.open(path).convert("L") 
        img = img.resize((IMSIZE, IMSIZE)) 

        T_img = torch.Tensor( ( np.array(img) /128) -1 ).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            preds = model.net_G(T_img.unsqueeze(0).to(device))
        colorized = lab_to_rgb(T_img.unsqueeze(0), preds.cpu())[0]
        img = Image.fromarray( (colorized*255).astype(np.uint8) )
        img.save(out_png)
        if wld.exists(): 
            shutil.copy(wld, OUT_DIR )
