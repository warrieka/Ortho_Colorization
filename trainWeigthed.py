from pathlib import Path
import torch, datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.tools import create_loss_meters, update_losses, log_results, visualize
from model.unet import ResUnet
from model.pil_Dataset import makeWeightedDatasetFromFeather, ColorizationDataset #, makeSubsetsFromList
from model.mainModel import MainModel
# multi-GPU support  
from accelerate import Accelerator

# HYPERPARAMETERS
IMAGE_SIZE = 512
EPOCHS =     16
TRAIN_DS_SIZE = 50000
TEST_DS_SIZE  = 400
LR_GENERATOR =     5e-4
LR_DISCRIMINATOR = 5e-4
FEATHER_DS =      Path('.\\data\\tiles_2015_Weighted.arrow').resolve()
DS_PATH_FIELD =   'path'
DS_WEIGHT_FIELD = 'WEIGHT'
PRETRAINED_DICT = Path(f".\\runs\\pretrain\\resnet18_0.60m_224_run0.pth").resolve()
OUT_STATE_DICT =  Path(f'.\\runs\\models\\run26\\color_run26_{IMAGE_SIZE}.pth').resolve()
RESUME = Path(f'.\\runs\\models\\run26\\color_run26_512_epoch12.pth').resolve()
START_EPOCH = 12

def train_model(train_dl, test_dl, epochs, start=0):
    accelerator = Accelerator(mixed_precision='fp16', project_dir=OUT_STATE_DICT.parent)
    train_dl, test_dl = accelerator.prepare(train_dl, test_dl)

    if PRETRAINED_DICT is not None and PRETRAINED_DICT.exists():
        net_G = ResUnet(n_input=1, n_output=2)
        net_G.load_state_dict(torch.load(PRETRAINED_DICT))
    else:
        net_G = None

    model = MainModel(lr_G=LR_GENERATOR, lr_D=LR_DISCRIMINATOR, 
                      net_G=net_G, accelerator=accelerator)
    if RESUME:
        model.load_state_dict( torch.load( RESUME ) )

    # getting a batch of test data for visualizing the model output after an epoch
    test_data = iter(test_dl) 
    print(f"Training started at {datetime.datetime.now()}")

    logfile = OUT_STATE_DICT.parent / f"{OUT_STATE_DICT.stem}.csv"
    if start == 0:
        with open(logfile , 'w' ) as log:
            log.write( ";".join(create_loss_meters()) +"\n")
        
    for e in range(start,epochs):
        # function returing a dictionary of objects to log the losses of the complete network
        loss_meter_dict = create_loss_meters()
        for train_data in tqdm(train_dl):
            model.setup_input(train_data) 
            model.optimize()
            # function updating the log objects
            update_losses(model, loss_meter_dict, count=train_data['L'].size(0)) 

        print(f"\nEpoch {e+1}/{EPOCHS} [{datetime.datetime.now()}]")
        # function to print out the losses
        log_results(loss_meter_dict, logFile= logfile )
        # function displaying the model's outputs
        visualize(model, next(test_data), epoch=e, save_dir= OUT_STATE_DICT.parent )
        # save intemediatie results 
        accelerator.wait_for_everyone()
        # accelerator.save_state(OUT_STATE_DICT.parent)
        torch.save(model.state_dict(), OUT_STATE_DICT.parent / f"{OUT_STATE_DICT.stem}_epoch{e+1}.pth" )

    print(f"Training ended at {datetime.datetime.now()}")
    torch.save(model.state_dict(), OUT_STATE_DICT)

def main(): 
    train_paths= makeWeightedDatasetFromFeather( arrow=FEATHER_DS, 
            pathField=DS_PATH_FIELD, weigthField=DS_WEIGHT_FIELD, size=TRAIN_DS_SIZE)
    test_paths = makeWeightedDatasetFromFeather( arrow=FEATHER_DS, 
            pathField=DS_PATH_FIELD, weigthField=DS_WEIGHT_FIELD, size=TEST_DS_SIZE)
    # data =  list( Path(r"W:\testdata\tiles_2022_rbg").glob("*.png") )
    # train_paths, test_paths = makeSubsetsFromList( data, 5000, 400, True )

    train_dl = DataLoader(ColorizationDataset(train_paths, imsize=IMAGE_SIZE), 
                        num_workers=4, pin_memory=True, batch_size=8)

    test_dl = DataLoader(ColorizationDataset(test_paths, imsize=IMAGE_SIZE), 
                        num_workers=2, pin_memory=True, batch_size=4)

    train_model(train_dl, test_dl, EPOCHS, START_EPOCH)

if __name__ == "__main__":
    main()