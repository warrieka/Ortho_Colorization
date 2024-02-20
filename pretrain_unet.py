from pathlib import Path
import datetime, argparse
from torch.utils.data import DataLoader
from model.unet import ResUnet, pretrain_generator
from model.gdal_Dataset import arrowDataset

# HYPERPARAMETERS
IMAGE_SIZE = 512
MODEL_ARCHITECTURE = 'resnet50'
EPOCHS =     20
LR= 5e-4
PRETRAIN_DS_SIZE = 2000
FEATHER_DS = Path('.\\data\\tiles_merged.arrow').resolve()
DS_PATH_FIELD = 'path'
DS_WEIGHT_FIELD = 'WEIGHT'
PRETRAINED_DICT = Path(f".\\runs\\pretrain\\{MODEL_ARCHITECTURE}_{IMAGE_SIZE}.pth").resolve()
RESUME = False

def main(opts:dict): 
    
    pretrain_ds = arrowDataset(arrow=opts['dataset'], imsize=opts['imsize'], 
                            pathField=opts["dataset_path_field"], weightField=opts['dataset_weight_field'],
                            count=opts["train_size"])
    pretrain_dl = DataLoader(pretrain_ds, pin_memory=True, num_workers=6, batch_size=2 )

    net_G = ResUnet(n_input=1, n_output=2, size=opts['imsize'], timm_model_name=opts["architecture"])

    pretrain_generator(net_G=net_G, pretrain_dl=pretrain_dl, epochs=opts['epochs'], 
                       lrate=opts['lr'], stateDict=opts['output_pretrained_weights'],
                       resumeWeigths= opts['output_pretrained_weights'].parent if RESUME else None )

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description=
      'Deeplearing model to colorize black&white orthophoto\'s.\nThis tool pretrains the model.')
    parser.add_argument('--imsize', default=IMAGE_SIZE, type=int, 
                    help='The size the input image will be resize to.')
    parser.add_argument('--epochs', default=EPOCHS, type=int, 
                    help='The number of epochs to train for.')
    parser.add_argument('--train_size', default=PRETRAIN_DS_SIZE, type=int, 
                    help='The number of images to load from the training data')
    parser.add_argument('--lr', default=LR, type=float, 
                    help='Learing rate of the generator.')
    parser.add_argument('--dataset',  default=FEATHER_DS, type=Path, 
                    help='Input traindata in Apache feather/arrow format.')
    parser.add_argument('--dataset_path_field', default=DS_PATH_FIELD, 
                    help='Fieldname to the path to the image.')
    parser.add_argument('--dataset_weight_field', default=DS_WEIGHT_FIELD, 
                    help='Fieldname to the weight of the image')
    parser.add_argument('--output_pretrained_weights', default=PRETRAINED_DICT, type=Path, 
                    help='File that contains the pretrained weights.')
    parser.add_argument('--architecture', default=MODEL_ARCHITECTURE,  
                    help='The architecture of the UNET, for example "resnet18" ')
    
    print(f"Started pretraining at {datetime.datetime.now()}")
    opts = vars( parser.parse_args() )
    main(opts)
    print(f"Finished pretraining at {datetime.datetime.now()}")