from pathlib import Path
import torch, datetime
from torch.utils.data import DataLoader
from model.unet import ResUnet, pretrain_generator
from model.pil_Dataset import makeWeightedDatasetFromFeather, ColorizationDataset


# HYPERPARAMETERS
IMAGE_SIZE = 224
EPOCHS =     10
PRETRAIN_DS_SIZE = 2000
FEATHER_DS =      Path('.\\data\\tiles_2015_Weighted.arrow').resolve()
DS_PATH_FIELD =   'path'
DS_WEIGHT_FIELD = 'WEIGHT'
PRETRAINED_DICT = Path(f".\\runs\\pretrain\\resnet18_0.60m_{IMAGE_SIZE}_pretrained.pth").resolve()

def main(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_paths= makeWeightedDatasetFromFeather( arrow=FEATHER_DS, 
            pathField=DS_PATH_FIELD, weigthField=DS_WEIGHT_FIELD, size=PRETRAIN_DS_SIZE)
    pretrain_dl = DataLoader(ColorizationDataset(pretrain_paths, imsize=EPOCHS),
                             num_workers=2, batch_size=8)

    net_G = ResUnet(n_input=1, n_output=2, size=IMAGE_SIZE)
    pretrain_generator(net_G=net_G, pretrain_dl=pretrain_dl)
    if PRETRAINED_DICT.exists():
        net_G.load_state_dict(torch.load(PRETRAINED_DICT, map_location=device))

    net_G= pretrain_generator(net_G=net_G, pretrain_dl=pretrain_dl)
    with torch.inference_mode():
        torch.save(net_G.state_dict(), PRETRAINED_DICT)

if __name__ == "__main__":
    print(f"Started pretraining at {datetime.datetime.now()}")
    main()
    print(f"Finished pretraining at {datetime.datetime.now()}")