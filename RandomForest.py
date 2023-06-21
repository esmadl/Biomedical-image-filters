#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 23:28:06 2023
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


# LBP Filters 
kernels = []
method = 'uniform'

#image_folder = "crops/train_image_crops"
#mask_folder = "crops/train_mask_crops"
image_folder = "membrane/train/image"
mask_folder = "membrane/train/label"


image_files = [file for file in os.listdir(image_folder) if not file.endswith('.DS_Store')]
image_files2 = [file for file in os.listdir(mask_folder) if not file.endswith('.DS_Store')]

image_files = sorted(image_files)
image_files2 = sorted(image_files2)

image_width = 100
image_height = 100

df = pd.DataFrame()

for image_file,mask_file in zip(image_files,image_files2):
    
    print(image_file,mask_file)
    
    image_path = os.path.join(image_folder, image_file)
    mask_path = os.path.join(mask_folder, mask_file) 

    
    print(image_path,mask_path)
    image = cv2.imread(image_path)  
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(mask_path)  
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

  
    resized_image = cv2.resize(image, (image_width, image_height))
    resized_mask = cv2.resize(mask, (image_width, image_height))
    
    
    img2 = image.ravel()
    df['Original Image'] = img2
    
    num =0
    for radius in range(1,10):   
        for n_point in range(1):  
            n_points =  radius * 8
            lbp_label = 'LBP' + str(num)  
            kernel_lbp = local_binary_pattern(image, n_points, radius, method)   
            kernels.append(kernel_lbp)
       
            fimg = cv2.filter2D(image, cv2.CV_8UC3, kernel_lbp)
            #cv2.imshow("filter",fimg)
            #cv2.waitKey(0)
    
            filtered_img = fimg.reshape(-1)
            df[lbp_label] = filtered_img 
            #print(lbp_label, ': radius=', radius, ': n_points=', n_points, ': method=', method)
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
                   # print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
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
            #print(glcm_label, ': distance=', distance, ': angle=', np.pi/angle*2, np.pi/angle, 3*np.pi/angle*2)     
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
    df['Variance s3'] = variance_img1  #Add column to original dataframe
    
    ##############
    
  
    labeled_img1 = mask.reshape(-1)
    df['Labels'] = labeled_img1

###
Y = df["Labels"].values

#columns=["Gaussian s3","Gaussian s7","Gabor24","Gabor21","Gabor23","Median s3","Gabor29","Gabor31","Gabor30","Gabor5"]

#X = df.loc[:,columns]

X = df.drop(labels = ["Labels"], axis=1) 


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)


from sklearn.ensemble import RandomForestClassifier
# Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators = 100, random_state = 42)

model.fit(X_train, y_train)


prediction_test_train = model.predict(X_train)

#Test prediction on testing data. 
prediction_test = model.predict(X_test)

from sklearn import metrics
#Print the prediction accuracy

#First check the accuracy on training data. This will be higher than test data prediction accuracy.
print ("Accuracy on training data = ", metrics.accuracy_score(y_train, prediction_test_train))
#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))


from sklearn import metrics
#Print the prediction accuracy

#First check the accuracy on training data. This will be higher than test data prediction accuracy.
print ("Accuracy on training data = ", metrics.accuracy_score(y_train, prediction_test_train))
#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))


feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp[:20])



import pickle

#Save the trained model as pickle string to disk for future use
filename = "rf_model"
pickle.dump(model, open(filename, 'wb'))

####################################################

end_time = time.time()

time = end_time-start_time
print("Trainin time: ", time)