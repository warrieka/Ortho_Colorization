from pathlib import Path
import datetime
from torch.utils.data import DataLoader
from model.unet import ResUnet, pretrain_generator
from model.pil_Dataset import makeWeightedDatasetFromFeather, ColorizationDataset

# HYPERPARAMETERS
IMAGE_SIZE = 256
MODEL_ARCHITECTURE = 'resnet34'
EPOCHS =     20
LR= 5e-4
PRETRAIN_DS_SIZE = 20000
FEATHER_DS = Path('.\\data\\tiles_2015_Weighted.arrow').resolve()
DS_PATH_FIELD = 'path'
DS_WEIGHT_FIELD = 'WEIGHT'
PRETRAINED_DICT = Path(f".\\runs\\pretrain\\{MODEL_ARCHITECTURE}_{IMAGE_SIZE}_run7.pth").resolve()
RESUME_DICT = None # Path(".\\runs\pretrain").resolve()

def main(): 
    pretrain_paths= makeWeightedDatasetFromFeather( arrow=FEATHER_DS, 
            pathField=DS_PATH_FIELD, weightField=DS_WEIGHT_FIELD, size=PRETRAIN_DS_SIZE)
    pretrain_dl = DataLoader(ColorizationDataset(pretrain_paths, imsize=IMAGE_SIZE, resize=True, grainify=False),
                            pin_memory=True, num_workers=4, batch_size=8 )

    net_G = ResUnet(n_input=1, n_output=2, size=IMAGE_SIZE, timm_model_name=MODEL_ARCHITECTURE)

    pretrain_generator(net_G=net_G, pretrain_dl=pretrain_dl, epochs=EPOCHS, 
                 lrate=LR, stateDict=PRETRAINED_DICT, resumeWeigths=RESUME_DICT)

if __name__ == "__main__":
    print(f"Started pretraining at {datetime.datetime.now()}")
    main()
    print(f"Finished pretraining at {datetime.datetime.now()}")