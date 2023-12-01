from pathlib import Path

import torch, datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.tools import create_loss_meters, update_losses, log_results, visualize
from model.pil_Dataset import makeWeightedDatasetFromFeather, ColorizationDataset
from model.mainModel import MainModel

# HYPERPARAMETERS
IMAGE_SIZE = 512
EPOCHS =     20
TRAIN_DS_SIZE = 12000
TEST_DS_SIZE  = 3000
LR_GENERATOR =     1e-4
LR_DISCRIMINATOR = 1e-4
IMG_ROOT =       r'W:\2023_1m_tiles'
FEATHER_DS =     r"W:\2023_1m_tiles\tile2023_weigths.arrow"
DS_PATH_FIELD = 'location'
DS_WEIGHT_FIELD = 'W'
OUT_STATE_DICT =  r'.\runs\model_pil_run10_1m.pth'


# # Train
def train_model(model, train_dl, test_dl, epochs):
    test_data = iter(test_dl) # getting a batch of test data for visualizing the model output after fixed intrvals
    print(f"Training started at {datetime.datetime.now()}")
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to log the losses of the complete network
        for train_data in tqdm(train_dl):
            model.setup_input(train_data) 
            model.optimize()
            # function updating the log objects
            update_losses(model, loss_meter_dict, count=train_data['L'].size(0)) 

        print(f"\nEpoch {e+1}/{EPOCHS} [{datetime.datetime.now()}]")
        log_results(loss_meter_dict) # function to print out the losses
        visualize(model, next(test_data), epoch=e, save_dir='runs/img') # function displaying the model's outputs
        with torch.inference_mode():
            outDict = Path(OUT_STATE_DICT)
            torch.save(model.state_dict(), f"runs/{outDict.stem}_epoch{e+1}.pth" )


def main(): 
    train_paths, test_paths = makeWeightedDatasetFromFeather(arrow=FEATHER_DS,  replacement=False,
                                            pathField=DS_PATH_FIELD, weigthField=DS_WEIGHT_FIELD,
                                            train_size=TRAIN_DS_SIZE, test_size=TEST_DS_SIZE, )

    train_dl = DataLoader(ColorizationDataset(train_paths, imsize=IMAGE_SIZE, rootDir=IMG_ROOT), 
                        num_workers=4, pin_memory=True, batch_size=16)
    test_dl = DataLoader(ColorizationDataset(test_paths, imsize=IMAGE_SIZE, rootDir=IMG_ROOT), 
                        num_workers=4, pin_memory=True, batch_size=8)
    
    model = MainModel(lr_G=LR_GENERATOR, lr_D=LR_DISCRIMINATOR)
    train_model(model, train_dl, test_dl, EPOCHS)
    model.eval()
    with torch.inference_mode():
        torch.save(model.state_dict(), OUT_STATE_DICT)

if __name__ == "__main__":
    main()