# NISRec
This is our Pytorch implementation for the paper:
The code is tested under a Linux desktop (RTX 3090 GPU) with Pytorch 1.7 and Python 3.6.
#Dependencies
Pytorch >= 1.6
Python >= 3.5
numpy
# Model Training
To train our model on the dataset (eg ml-1m) with default hyper-parameters:
python main.py --dataset=ml-1m --train_dir=default 
