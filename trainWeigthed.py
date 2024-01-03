from pathlib import Path
import torch, datetime, argparse
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
EPOCHS =     20
START_EPOCH = 0
TRAIN_DS_SIZE = 50000
LR_GENERATOR =     5e-4
LR_DISCRIMINATOR = 5e-4
FEATHER_DS =      Path('.\\data\\tiles_2015_Weighted.arrow')
DS_PATH_FIELD =   'path'
DS_WEIGHT_FIELD = 'WEIGHT'
PRETRAINED_DICT = Path(f".\\runs\\pretrain\\resnet18_0.60m_224_run0.pth")
OUT_STATE_DICT =  Path(f'.\\runs\\models\\run26\\color_run26_{IMAGE_SIZE}.pth')
RESUME = None #Path(f'.\\runs\\models\\run26\\color_run26_512_epoch20.pth')


def train_model(train_dl, test_dl, opts):

    proj_dir = opts.output_weights.parent.resolve()
    pretrained = opts.pretrained_weights.resolve() if opts.pretrained_weights else None

    accelerator = Accelerator(mixed_precision='fp16', project_dir=proj_dir)
    train_dl, test_dl = accelerator.prepare(train_dl, test_dl)

    if pretrained and pretrained.exists():
        net_G = ResUnet(n_input=1, n_output=2)
        net_G.load_state_dict(torch.load(pretrained))
    else:
        net_G = None

    model = MainModel(lr_G=opts.lr_net_G, lr_D=opts.lr_net_D, 
                      net_G=net_G, accelerator=accelerator)
    if opts.resume_from:
        model.load_state_dict( torch.load( opts.resume_from ) )

    # getting a batch of test data for visualizing the model output after an epoch
    test_data = iter(test_dl) 
    print(f"Training started at {datetime.datetime.now()}")

    logfile = proj_dir / f"{opts.output_weights.stem}.csv"
    if opts.resume_epoch == 0:
        with open(logfile , 'w' ) as log:
            log.write( ";".join(create_loss_meters()) +"\n")
        
    for e in range(opts.resume_epoch, opts.epochs):
        # function returing a dictionary of objects to log the losses of the complete network
        loss_meter_dict = create_loss_meters()
        for train_data in tqdm(train_dl):
            model.setup_input(train_data) 
            model.optimize()
            # function updating the log objects
            update_losses(model, loss_meter_dict, count=train_data['L'].size(0)) 

        print(f"\nEpoch {e+1}/{opts.epochs} [{datetime.datetime.now()}]")
        # function to print out the losses
        log_results(loss_meter_dict, logFile= logfile )
        # function displaying the model's outputs
        visualize(model, next(test_data), epoch=e, save_dir=proj_dir )
        # save intemediatie results 
        accelerator.wait_for_everyone()
        # accelerator.save_state(OUT_STATE_DICT.parent)
        torch.save(model.state_dict(), proj_dir / f"{opts.output_weights.stem}_epoch{e+1}.pth" )

    print(f"Training ended at {datetime.datetime.now()}")
    torch.save(model.state_dict(), opts.output_weights)

def main(opts): 
    test_ds_size  = opts.epochs *6
    train_paths= makeWeightedDatasetFromFeather( 
            arrow=opts.dataset.resolve(), size=opts.train_size,
            pathField=opts.dataset_path_field, weightField=opts.dataset_weight_field )
    test_paths = makeWeightedDatasetFromFeather( 
            arrow=opts.dataset.resolve(), size=test_ds_size,
            pathField=opts.dataset_path_field, weightField=opts.dataset_weight_field )

    train_dl = DataLoader(ColorizationDataset(train_paths, imsize=opts.imsize ), 
                        num_workers=4, pin_memory=True, batch_size=8)
    test_dl = DataLoader(ColorizationDataset(test_paths, imsize=opts.imsize ), 
                        num_workers=2, pin_memory=True, batch_size=4)
    train_model(train_dl, test_dl, opts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description=
      'Deeplearing model to colorize black&white orthophoto\'s.\nThis tool trains the model.')
    parser.add_argument('--imsize', default=IMAGE_SIZE, type=int, 
                    help='The size the input image will be resize to.')
    parser.add_argument('--epochs', default=EPOCHS, type=int, 
                    help='The number of epochs to train for.')
    parser.add_argument('--train_size', default=TRAIN_DS_SIZE, type=int, 
                    help='The number of images to load from the training data')
    parser.add_argument('--lr_net_G', default=LR_GENERATOR, type=float, 
                    help='Learing rate of the generator.')
    parser.add_argument('--lr_net_D', default=LR_DISCRIMINATOR, type=float, 
                    help='Learing rate of the discriminator.')
    parser.add_argument('--dataset',  default=FEATHER_DS, type=Path, 
                    help='Input traindata in Apache feather/arrow format.')
    parser.add_argument('--dataset_path_field', default=DS_PATH_FIELD, 
                    help='Fieldname to the path to the image.')
    parser.add_argument('--dataset_weight_field', default=DS_WEIGHT_FIELD, 
                    help='Fieldname to the weight of the image')
    parser.add_argument('--pretrained_weights', default=PRETRAINED_DICT, type=Path, 
                    help='File that contains the pretrained weights.')
    parser.add_argument('--output_weights', default=OUT_STATE_DICT, type=Path, 
                    help='The filename and path to the output weights.')
    parser.add_argument('--resume_from', default=RESUME, type=Path, 
                    help='Resume traings from these weights')
    parser.add_argument('--resume_epoch', default=START_EPOCH, type=int, 
                    help='The epoch to resume from.')

    opts = parser.parse_args()
    main(opts)