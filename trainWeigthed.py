from pathlib import Path

import torch, datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.tools import create_loss_meters, update_losses, log_results, visualize
from model.unet import ResUnet
from model.pil_Dataset import makeWeightedDatasetFromFeather, ColorizationDataset
from model.mainModel import MainModel

# HYPERPARAMETERS
IMAGE_SIZE = 224
EPOCHS =     30
TRAIN_DS_SIZE = 16000
TEST_DS_SIZE  = 3000
LR_GENERATOR =     5e-5
LR_DISCRIMINATOR = 5e-5
FEATHER_DS =      Path('.\\data\\tiles_2015_Weighted.arrow').resolve()
DS_PATH_FIELD =   'path'
DS_WEIGHT_FIELD = 'WEIGHT'
PRETRAINED_DICT = Path(f".\\runs\\pretrain\\resnet18_0.60m_{IMAGE_SIZE}_pretrained.pth").resolve()
OUT_STATE_DICT =  Path(f'.\\runs\\models\\color_run18_0.60m_{IMAGE_SIZE}.pth').resolve()
RESUME_DICT = None #Path(".\\runs\\model_pil_run15_0.30m_512_epoch5.pth").resolve()


# # Train
def train_model(model, train_dl, test_dl, epochs):
    # getting a batch of test data for visualizing the model output after fixed intrvals
    test_data = iter(test_dl) 
    print(f"Training started at {datetime.datetime.now()}")
    
    for e in range(epochs):
        # function returing a dictionary of objects to log the losses of the complete network
        loss_meter_dict = create_loss_meters()
        for train_data in tqdm(train_dl):
            model.setup_input(train_data) 
            model.optimize()
            # function updating the log objects
            update_losses(model, loss_meter_dict, count=train_data['L'].size(0)) 

        print(f"\nEpoch {e+1}/{EPOCHS} [{datetime.datetime.now()}]")
        # function to print out the losses
        log_results(loss_meter_dict)
        # function displaying the model's outputs
        visualize(model, next(test_data), epoch=e, save_dir='runs/img')
        with torch.inference_mode():
            outDict = Path(OUT_STATE_DICT)
            torch.save(model.state_dict(), f"runs/{outDict.stem}_epoch{e+1}.pth" )

    print(f"Training ended at {datetime.datetime.now()}")

def main(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_paths= makeWeightedDatasetFromFeather( arrow=FEATHER_DS, 
            pathField=DS_PATH_FIELD, weigthField=DS_WEIGHT_FIELD, size=TRAIN_DS_SIZE)
    test_paths = makeWeightedDatasetFromFeather( arrow=FEATHER_DS, 
            pathField=DS_PATH_FIELD, weigthField=DS_WEIGHT_FIELD, size=TEST_DS_SIZE)

    train_dl = DataLoader(ColorizationDataset(train_paths, imsize=IMAGE_SIZE), 
                        num_workers=4, pin_memory=True, batch_size=64)

    test_dl = DataLoader(ColorizationDataset(test_paths, imsize=IMAGE_SIZE), 
                        num_workers=2, pin_memory=True, batch_size=4)
    
    net_G = None
    if PRETRAINED_DICT.exists():
        net_G = ResUnet(n_input=1, n_output=2, size=256)
        net_G.load_state_dict(torch.load(PRETRAINED_DICT, map_location=device))

    model = MainModel(lr_G=LR_GENERATOR, lr_D=LR_DISCRIMINATOR, net_G=net_G)
    if RESUME_DICT is not None:
         model.load_state_dict( torch.load( RESUME_DICT , map_location=device ) )

    train_model(model, train_dl, test_dl, EPOCHS)
    model.eval()
    with torch.inference_mode():
        torch.save(model.state_dict(), OUT_STATE_DICT)

if __name__ == "__main__":
    main()