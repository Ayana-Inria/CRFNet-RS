# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:34:32 2023

@author: mpastori
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:04:59 2023

@author: mpastori
"""

import torch
from os import listdir
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from net.unet import *
import torch.optim as optim
from utils.export_result import *
from utils.utils_dataset import *
from net.net import Net
from dataset.dataset import ISPRS_dataset
from net.test_network import test


def main(args):

    input_folder = args.input 
    output_folder = args.output 
    
    
    # Parameters
    WINDOW_SIZE = args.window # Patch size
    IN_CHANNELS = 3 # Number of input channels (e.g. 3 for RGB), in this case the CSK image has 1 channel
    
    FOLDER = input_folder # "/path/to/the/dataset/folder/"
    batch_size = args.batch_size # Number of samples in a mini-batch
    epochs = args.epochs
    save_epoch = args.save_epoch
    
    labels = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
    N_CLASSES = len(labels) # Number of classes
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    CACHE = True # Store the dataset in-memory
    
    ################################################ CAMBIA
    DATA_FOLDER = FOLDER + '/top/top_mosaic_09cm_area{}.tif'
    LABEL_FOLDER = FOLDER + '/gt/top_mosaic_09cm_area{}.tif'
    ERODED_FOLDER = FOLDER + '/gt_eroded/top_mosaic_09cm_area{}_noBoundary.tif'
        
    net = CRFNet(n_channels=IN_CHANNELS, n_classes=N_CLASSES, bilinear=True)
    
    
    
    base_lr = args.base_lr
    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():
        if '_D' in key:
            # Decoder weights are trained at the nominal learning rate
            params += [{'params':[value],'lr': base_lr}]
        else:
            # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
            params += [{'params':[value],'lr': base_lr / 2}]
    
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    # We define the scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)

    # look for gpu existence
    if torch.cuda.is_available():
        net.cuda()
    

    train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
    test_ids = ['5', '15', '21', '30'] 
     
            
    print("Tiles for training : ", train_ids)
    print("Tiles for testing : ", test_ids)
    

    train_set = ISPRS_dataset(train_ids, ids_type='TRAIN', gt_type = args.gt_type, gt_modification = disk(args.ero_disk), data_files=DATA_FOLDER, label_files = LABEL_FOLDER, window_size=WINDOW_SIZE, cache=CACHE)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size)
    test_set = ISPRS_dataset(test_ids, ids_type='TEST', gt_type = args.gt_type, gt_modification = disk(args.ero_disk), data_files=DATA_FOLDER, label_files = LABEL_FOLDER, window_size=WINDOW_SIZE,cache=CACHE)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size)
    

    # if you need to retrain the network
    if args.retrain:
        from net.train import train
        train(net, optimizer, epochs, save_epoch, WEIGHTS, train_loader, batch_size, WINDOW_SIZE, output_folder, scheduler)

    else:
        model_weights = './output/test_final'  # PATH/TO/PRETRAINED/MODEL 
        net.load_state_dict(torch.load(model_weights))

    
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    stride = 32
    acc_test, all_preds, all_gts = test(net, test_ids, test_images, test_labels, eroded_labels, labels, stride, batch_size, window_size=WINDOW_SIZE, all=True)
        

    # --- STARTING THE EXPERIMENT
    # experiment_name identify which experiment to run.
    experiment_name = args.experiment_name 
    # outputs are placed in a folder named experiment_name in the output folder
    output_path = set_output_location(experiment_name, output_folder)
    
    ########### save results ##########
    title = ("Quantitative results of experiment " + experiment_name)
    export_results(all_preds, all_gts, output_folder, experiment_name, 
                    confusionMat=True,
                    prodAccuracy=True,
                    averageAccuracy=True,
                    kappaCoeff=True,
                    title=title)
    
    for pred, ids in zip(all_preds, test_ids):
        img = convert_to_color(pred)
        plt.imshow(img) and plt.show()
        io.imsave(output_path + '/segmentation_result_area{}.png'.format(ids), img)
        



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='CRFNet for RS semantic segmentation')
    arg_parser.add_argument('-i', '--input', help='Path of input directory',
                            metavar='INPUT_DIR_PATH', default="./input/")
    arg_parser.add_argument('-o', '--output', help='Path of output directory',
                            metavar='OUTPUT_DIR_PATH', default="./output/")
    arg_parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the neural network')
    arg_parser.add_argument('-w', '--window', default=(256, 256), type=tuple, nargs='?',
                            help='Dimension of the crops of the images input to the network. Default is ' + str((256, 256)), metavar='WINDOW_SIZE')
    arg_parser.add_argument('-b', '--batch_size', default=10, type=int, nargs='?',
                            help='Size of the image batch input to the network. Default is ' + str(10))    
    arg_parser.add_argument('-d', '--ero_disk', default=8, type=int, nargs='?',
                            help='Size of the morphological disk to perform erosion. Default is ' + str(8))
    arg_parser.add_argument('-g', '--gt_type', required=True, choices=['ero', 'full', 'conncomp'],
                            help='Type of GT used to train the network.')
    arg_parser.add_argument('-exp', '--experiment_name', default='tmp', type=str, nargs='?',
                            help='Experiment_name identify the name of the experiment folder.')
    arg_parser.add_argument('-lr', '--base_lr', default=0.01, type=float, help='Base learning rate of the neural network')
    arg_parser.add_argument('-e', '--epochs', default=30, type=int, help='Number of epochs of the training of the neural network')
    arg_parser.add_argument('-se', '--save_epoch', default=10, type=int, nargs='?',
                            help='When to save the model')


    args = arg_parser.parse_args()

    main(args)