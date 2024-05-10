Historical Panchromatic Orthophoto Colorisation with a Generative Adversarial Neural net
=================================================================
By Kay Warrie

This is deeplearing model to colorize historical greyscale or panchromatic orthophotos and orthophoto mosaics. 
Greyscale images that made using all the wavelengths of the visible spectrum are called **panchromatic**, most historical images are panchrommatic. 
An **Orthophoto** is an aerial photograph geometrically corrected ("orthorectified") such that the scale is uniform. It is the basis for most mapping solutions.
An **Orthophoto mosaic** is a type of large scale image that is created by stitching together a collection orthophoto to produce a seamless, georeferenced image, for example "Satelite"-view in google maps, that is by thet way, mostly made with aerial photo's and not with satelite images. 
A **Generative Adversarial Network (GAN)** is a type of artificial intelligence, a generative model consisting of two neural networks, the generator and the discriminator. 
The generator is convolutional neural net that makes an image and the discriminator is en model the tries to distinguish between the label data and the generated images. 
The Loss of GAN's discriminator calculated by passing the batch of the generators output and a batch of real data and seeing if it can distinguish between the two. The loss of generator is output of the discriminator.

A full explantation how this model was constructed can found in [explanation.ipynb](explanation.ipynb).

To train and run the final a series of commandline tools was constructed:

- [pretrain_unet.py](pretrain_unet.py) -> initialise the U-net generator.
- [trainWeigthed.py](trainWeigthed.py) -> train the full GAN.
- [inference.py](inference.py) -> test the model on real greyscale images.

To showcase the results a interactive webpage was constructed: <https://warrieka.github.io/histo_ortho_viewer>

Pretraining 
-----------

In order to initialise the weigths of the generator, it can be pretrained by running a few image througth the unet without the discriminator, this should improve training speed. 

I used the script [pretrain_unet.py](pretrain_unet.py). 

Run `python .\pretrain_unet.py` with the following options: 

    -h, --help              Show this help message and exit
    --imsize IMSIZE         The size the input image will be resize to.
    --epochs EPOCHS         The number of epochs to train for.
    --train_size TRAIN_SIZE
                            The number of images to load from the training data
    --lr LR                 Learing rate of the generator.
    --dataset DATASET       Input traindata in Apache feather/arrow format.
    --dataset_path_field DATASET_PATH_FIELD
                            Fieldname to the path to the image.
    --dataset_weight_field DATASET_WEIGHT_FIELD
                            Fieldname to the weight of the image
    --output_pretrained_weights OUTPUT_PRETRAINED_WEIGHTS
                            File that contains the pretrained weights.
    --architecture ARCHITECTURE
                            The architecture of the UNET, for example "resnet18"


Training
------

To train for production you can use `python .\trainWeigthed.py` to run the script with these options:

    -h, --help               show this help message and exit.
    --imsize IMSIZE          The size the input image will be resize to.
    --epochs EPOCHS          The number of epochs to train for.
    --train_size TRAIN_SIZE  The number of images to load from the training data.
    --lr_net_G LR_NET_G      Learing rate of the generator.
    --lr_net_D LR_NET_D      Learing rate of the discriminator.
    --dataset DATASET        Input traindata in Apache feather/arrow format.
    --dataset_path_field DATASET_PATH_FIELD
                             Fieldname to the path to the image.
    --dataset_weight_field DATASET_WEIGHT_FIELD
                             Fieldname to the weight of the image
    --pretrained_weights PRETRAINED_WEIGHTS
                             File that contains the pretrained weights.
    --output_weights OUTPUT_WEIGHTS
                             The filename and path to the output weights.
    --resume_from RESUME_FROM
                             Resume traings from these weights
    --resume_epoch RESUME_EPOCH
                             The epoch to resume from.

You can also change the values in CAPITAL-case in top of the script to your settings to change te default values. 

I trained the final model for 50 epoch's on 50000 images of 512x512 pixels with a ground resoltion between 0.3 and 1 meter. 

You can see the result for each epoch on this gif: 

![](pic/training.gif)


Inference and testing
---------------------

The script to test on real data is called `inference.py`. 
It allows tou to convert a GDAL-readable black&white source to colorized data. 
You can have multiple inputs by using glob expression (*.tif). 
Outputs are written to a [gdal-driver][8] that supports `create`, like Geotiff.
It preserves geospatial metadata, like crs and geotransform.  

run `python .\inference.py` 

Options:

    -h, --help          Show this help message and exit.
    --input INPUT       The input file(s), you can use a glob expression like "*.tif" to specify multiple files
    --out_dir OUT_DIR   The output location of the resulting colorisation, don't use the input folder!
    --out_driver OUT_DRIVER
                        The output gdal driver to use to write output, 
                        only drivers that support "create" can be used.
                        It defaults to Geotiff (see https://gdal.org/drivers/raster/)
    --batch_size BATCH_SIZE
                        the size of batch of tiles the algoritem sends to the GPU, 
                        If you get *CUDA out of memory issues*, try to decrease the batch.
                        It defaults to 12 
    --nodata NODATA     The pixel value to use for NODATA transparancy, defaults to 255 

