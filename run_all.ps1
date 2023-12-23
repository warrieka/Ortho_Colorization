conda activate fastai
Set-Location V:\project\learn\Ortho_Colorization
python .\pretrain_unet.py 
python .\trainWeigthed.py
python .\trainGDAL.py