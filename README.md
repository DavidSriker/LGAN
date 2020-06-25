# LGAN
This project is part of the course "Deep Learning in Medical Imaging" in Tel-Aviv University.

This repo contains unofficial Pytorch implementation of segmenting images using deep learning network based on the published paper: [LGAN: Lung Segmentation in CT Scans Using Generative Adversarial Network](https://arxiv.org/abs/1901.03473)

This was built for a binary segmentaion task, but can easily be extended to multi-class segmentation.

The full report and original paper can be seen in the "paper_and_report" directory.

## Installation

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Requirements

This code was tested with:
* Ubuntu 18.04 with python 3.6.9

### Step-by-Step Procedure
In order to set the virtual environment use the following commands to create a new working environment with all the required dependencies.

**GPU based enviroment**:
```
git clone https://github.com/DavidSriker/LGAN
cd LGAN
pip install -r pip_requirements.txt
```

*In order to utilize the GPU implementation make sure your hardware and operation system are compatible for Pytorch with python 3*

### Donwload Data
Download the Kaggle challenge datasets:
1. [Lung Data](https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels)
2. [Prostate Data](https://www.kaggle.com/aaryapatel98/prostate)

using the bash script in this repo in the following manner (you should make sure that `unzip` is present in your machine):
```
./DataCollect.sh
```

## Training the model

Before training the model, possible modification can be done. Run the following to see the available parameters to change:

```
python3 Train.py -h
```

Then you can either run the script with the defaults or with your own parameters in the following way:

1. Default:
    ```
    python3 Train.py
    ```
2. Own parameters:
    ```
    python3 Train.py --batch_size 16 --dataset_name prostate --n_epochs 200
    ```

## Testing the model

Before testing the model, the train procedure need to run in order to create the splitting (train/test/validation)
and to create the models.
possible modification can be done. Run the following to see the available parameters to change:

```
python3 Test.py -h
```

Then you can either run the script with the defaults or with your own parameters in the following way:

1. Default:
    ```
    python3 Test.py
    ```
2. Own parameters:
    ```
    python3 Test.py --dataset_name prostate --n_epochs 100
    ```


## Authors
* **David Sriker** - *David.Sriker@gmail.com*
