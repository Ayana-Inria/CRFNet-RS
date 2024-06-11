# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 19:29:23 2022

@author: marti
"""

import numpy as np
from skimage import io
from utils.utils_dataset import *
from tqdm import tqdm
from utils.utils import *
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


def test(net, test_ids, test_images, test_labels, eroded_labels, classes, stride, batch_size, window_size, all=False):

    all_preds = []
    all_gts = []
    
    # Switch the network to inference mode
    net.eval()

    for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (len(classes),))
        
        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):
            # Display in progress results
            
            if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                    _pred = np.argmax(pred, axis=-1)
                    fig = plt.figure()
                    fig.add_subplot(1,3,1)
                    plt.imshow(np.asarray(255 * img, dtype='uint8'))
                    fig.add_subplot(1,3,2)
                    plt.imshow(convert_to_color(_pred))
                    fig.add_subplot(1,3,3)
                    plt.imshow(gt)
                    plt.show()
                    
            # Build the tensor
            image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
            
            # Do the inference
            outs = net(image_patches)[0]
            outs = outs.data.cpu().numpy()
             
            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1,2,0))
                pred[x:x+w, y:y+h] += out
              
            del(outs)
        pred = np.argmax(pred, axis=-1)

        # Display the result
        
        fig = plt.figure()
        fig.add_subplot(1,3,1)
        plt.imshow(np.asarray(255 * img, dtype='uint8'))
        fig.add_subplot(1,3,2)
        plt.imshow(convert_to_color(pred))
        fig.add_subplot(1,3,3)
        plt.imshow(gt)
        plt.show()
        
        all_preds.append(pred)
        all_gts.append(gt_e)
        
        # Compute some metrics
        metrics(pred.ravel(), gt_e.ravel(), classes)
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel(), classes)
        
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy