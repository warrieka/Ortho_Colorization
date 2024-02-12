FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

RUN mkdir /colorisation
WORKDIR /colorisation

COPY ./requirements.txt /colorisation/requirements.txt
RUN pip install -r /colorisation/requirements.txt

RUN mkdir /colorisation/data
RUN mkdir /colorisation/runs
COPY ./model /colorisation/model
COPY ./pretrain_unet.py /colorisation/pretrain_unet.py
COPY ./trainWeigthed.py /colorisation/trainWeigthed.py
