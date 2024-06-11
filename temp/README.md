# FCN and Fully Connected NN for Remote Sensing Image Classification

This repository contains the code related to the paper:  

M. Pastorino, G. Moser, S. B. Serpico, and J. Zerubia, "Fully convolutional and feedforward networks for the semantic segmentation of remotely sensed images," 2022 IEEE International Conference on Image Processing, 2022, [https://hal.inria.fr/hal-03720693](https://hal.inria.fr/hal-03720693).

When using this work, please cite our IEEE ICIP'22 conference paper:

M. Pastorino, G. Moser, S. B. Serpico, and J. Zerubia, "Fully convolutional and feedforward networks for the semantic segmentation of remotely sensed images," in IEEE International Conference on Image Processing, Bordeaux, France, 2022. 

```
@ARTICLE{pastorino_icip22,
  author={Pastorino, Martina and Moser, Gabriele and Serpico, Sebastiano B. and Zerubia, Josiane},
  journal={IEEE International Conference on Image Processing}, 
  title={Fully convolutional and feedforward networks for the semantic segmentation of remotely sensed images}, 
  year={2022},
  volume={},
  number={},
  pages={},
  doi={}}
```

## Installation

The code was built on a virtual environment running on Python 3.9

### Step 1: Clone the repository

```
git clone --recursive https://github.com/Ayana-Inria/FCN-FFNET_RS-semantic-segmentation.git
```

### Step 2: Clone the repository

```
cd FCN-FFNET_RS-semantic-segmentation

pip install -r requirements.txt
```

### Step 3: Run the code

1. Train the model on a scarce GT set 

```
python main.py -r -g conncomp
```
2. Infer on data

```
python main.py -g full
```


## Project structure

```
semantic_segmentation
├── dataset - contains the data loader
├── input - images to train and test the network 
├── net - contains the loss, the network, and the training and testing functions
├── output - should contain the results of the training / inference
|   ├── exp_name
|   └── model.pth
├── utils - misc functions
└── main.py - program to run
```
  
## Data

The model is trained on the [ISPRS Vaihingen dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) and [ISPRS Potsdam dataset](http://www2.isprs.org/potsdam-2d-semantic-labeling.html). The two datasets consist of VHR optical images (spatial resolutions of 9 and 5cm, respectively), we used the IRRG channels. They can be downloaded on [Kaggle](https://www.kaggle.com/datasets/bkfateam/potsdamvaihingen) and should be inserted in the folder `/input`.

The data should have the following structure. 

```
input
├── top
|   └── top_mosaic_09cm_area{}.tif
├── gt
|   └── top_mosaic_09cm_area{}.tif
└── gt_eroded
    └── top_mosaic_09cm_area{}_noBoundary.tif
```


## License

The code is released under the GPL-3.0-only license. See `LICENSE.md` for more details.

## Acknowledgements

This work was conducted during my joint PhD at [INRIA](https://team.inria.fr/ayana/team-members/), d'Université Côte d'Azur and at the [University of Genoa](http://phd-stiet.diten.unige.it/). 
The ISPRS 2D Semantic Labeling Challenge Datasets were provided by the German Society for Photogrammetry, Remote Sensing and Geoinformation (DGPF).
The code to deal with the ISPRS dataset derives from the GitHub repository [Deep learning for Earth Observation](https://github.com/nshaud/DeepNetsForEO).
