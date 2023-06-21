#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 15:37:49 2023

@author: esmanur
"""

import cv2
import numpy as np
import pandas as pd

from skimage.feature import local_binary_pattern
from skimage.color import label2rgb
from skimage.transform import rotate
import matplotlib.pyplot as plt
import time
from skimage.feature import graycomatrix, graycoprops
from skimage import data
import os


start_time = time.time() 


kernels = []
method = 'uniform'



image_width = 100
image_height = 100

df = pd.DataFrame()

def feature_extraction(image):
    img2 = image.ravel()
    df['Original Image'] = img2
    
    num =0
    for radius in range(1,10):   
        for n_point in range(1):  
            n_points =  radius * 8
            lbp_label = 'LBP' + str(num)  
            kernel_lbp = local_binary_pattern(image, n_points, radius, method)   
            kernels.append(kernel_lbp)
            #Now filter the image and add values to a new column 
            fimg = cv2.filter2D(image, cv2.CV_8UC3, kernel_lbp)
            #cv2.imshow("filter",fimg)
            #cv2.waitKey(0)
    
            filtered_img = fimg.reshape(-1)
            df[lbp_label] = filtered_img 
            print(lbp_label, ': radius=', radius, ': n_points=', n_points, ': method=', method)
            num += 1  
    
    num_g = 1
    kernels_g = []
    for theta in range(3):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.2,0.1):
    #               print(theta, sigma, , lamda, frequency)
    
                    gabor_label = 'Gabor' + str(num_g)
    #                    print(gabor_label)
                    ksize=9
                    kernel_g = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels_g.append(kernel_g)
    
                    #Now filter image and add values to new column
                    fimg_g = cv2.filter2D(image, cv2.CV_8UC3, kernel_g)
    
                    #cv2.imshow("filter",fimg_g)
                    #cv2.waitKey(0)
                    filtered_img_g = fimg_g.reshape(-1)
    
                    df[gabor_label] = filtered_img_g  #Modify this to add new column for each gabor
                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num_g += 1        
    
    num_glcm=0
    for distance in range(1,2):
        for angle in range(2,4):
            glcm_label = 'GLCM' + str(num_glcm)
            # select some patches from grassy areas of the image
            glcm = graycomatrix(image, [distance], [0, np.pi/angle*2, np.pi/angle, 3*np.pi/angle*2])
    
            diss = graycoprops(glcm, 'dissimilarity')[0, 0]
            corr = graycoprops(glcm, 'correlation')[0, 0]
            df[glcm_label] = diss
            df[glcm_label] = corr
            print(glcm_label, ': distance=', distance, ': angle=', np.pi/angle*2, np.pi/angle, 3*np.pi/angle*2)     
            num_glcm += 1
    
    
    #CANNY EDGE
    edges = cv2.Canny(image, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe
    
    from skimage.filters import roberts, sobel, scharr, prewitt
    
    #ROBERTS EDGE
    edge_roberts = roberts(image)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    
    #SOBEL
    edge_sobel = sobel(image)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    
    #SCHARR
    edge_scharr = scharr(image)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1
    
    #PREWITT
    edge_prewitt = prewitt(image)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    
    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(image, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1
    
    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(image, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    
    #MEDIAN with sigma=3
    median_img = nd.median_filter(image, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    
    #VARIANCE with size=3
    variance_img = nd.generic_filter(image, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1
                
            
    return df


#########################################################

#Applying trained model to segment multiple files. 

import glob
import pickle
from matplotlib import pyplot as plt

filename = "rf_model"
loaded_model = pickle.load(open(filename, 'rb'))

path = "membrane/test/image/img.png"
#path = "crops/test_image_crops/img_crop_0_0.jpg"
i=0
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    img1= cv2.imread(file)
    img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#Call the feature extraction function.
    X = feature_extraction(img)
    
    result = loaded_model.predict(X)
    segmented = result.reshape((img.shape))
    
    name = file.split("e_")
    #plt.imsave('images/'+ name[1], segmented, cmap ='jet')
    
    from matplotlib import pyplot as plt
    plt.subplot(211)
    plt.imshow(img,cmap="gray")
    plt.subplot(212)
    plt.imshow(segmented, cmap="gray")
    plt.imsave('segmented/segmentedrf'+str(i)+'.jpg', segmented,cmap="gray")
    i+=1
    
    

#pre_img = "segmented/segmentedsplitrf0.jpg"
#real_img= "crops/test_mask_crops/mask_crop_0_0.jpg"

pre_img = "segmented/segmentedrf0.jpg"
real_img= "membrane/test/mask/mask.png"

def iou_score(y_true, y_pred, smooth=1):
    # Convert the input arrays to binary images (0 or 255)
    
    y_true = cv2.imread(y_true, cv2.IMREAD_GRAYSCALE)
    y_pred = cv2.imread(y_pred, cv2.IMREAD_GRAYSCALE)
    
    width, height = 256, 256
    y_true = cv2.resize(y_true, (width, height))
    y_pred = cv2.resize(y_pred, (width, height)) 
    
    y_true = np.uint8(y_true > 0.5) * 255
    y_pred = np.uint8(y_pred > 0.5) * 255

    # Compute the intersection and union using OpenCV functions
    intersection = cv2.bitwise_and(y_true, y_pred)
    union = cv2.bitwise_or(y_true, y_pred)

    # Calculate the IOU score
    iou = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)

    return iou

score = iou_score(real_img,pre_img)

print("IoU score: ",score)