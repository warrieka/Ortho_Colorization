from pathlib import Path
import torch, datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.tools import create_loss_meters, update_losses, log_results, visualize
from model.gdal_Dataset import gdalTrainDataset, gdalTestDataset
from model.mainModel import MainModel
from model.unet import ResUnet

# HYPERPARAMETERS
TILE_SIZE = 512
EPOCHS =     10
START_EPOCH = 10
LR_GENERATOR =     1e-4
LR_DISCRIMINATOR = 1e-4
IMG_TRAIN =  r"W:\testdata\train2023.tif"
IMG_TEST =   r"W:\testdata\test2023.tif"
PRETRAINED_DICT = r'.\runs\model_pil_run10_1m_512.pth'
OUT_STATE_DICT =  r'.\runs\model_gdal_run8_0.15m_512.pth'

# # Train
def train_model(model, train_dl, test_dl):
    test_data = iter(test_dl) # getting a batch of test data for visualizing the model output after fixed intrvals
    print(f"Training started at {datetime.datetime.now()}")
    for e in range(START_EPOCH,EPOCHS+START_EPOCH):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to log the losses of the complete network
        for train_data in tqdm(train_dl):
            model.setup_input(train_data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=train_data['L'].size(0)) # function updating the log objects

        print(f"\nEpoch {e+1}/{EPOCHS} [{datetime.datetime.now()}]")
        log_results(loss_meter_dict) # function to print out the losses
        visualize(model, next(test_data), epoch=e, save_dir='runs/img') # function displaying the model's outputs

        outDict = Path(OUT_STATE_DICT)
        torch.save(model.state_dict(), f"runs/{outDict.stem}_epoch{e+1}.pth" )

def main(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dl= DataLoader(gdalTrainDataset(IMG_TRAIN, imsize=TILE_SIZE), batch_size=64, num_workers=8)
    test_dl = DataLoader(gdalTestDataset( IMG_TEST , imsize=TILE_SIZE), batch_size=6, num_workers=2)

    model = MainModel(lr_G=LR_GENERATOR, lr_D=LR_DISCRIMINATOR, net_G=ResUnet())
    
    if PRETRAINED_DICT is not None:
         model.load_state_dict(
            torch.load( PRETRAINED_DICT , map_location=device ) )

    train_model(model, train_dl, test_dl)
    model.eval()
    with torch.inference_mode():
        torch.save(model.state_dict(), OUT_STATE_DICT)

if __name__ == "__main__":
    main()