#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:55:56 2023

@author: esmanur
"""

import cv2
import numpy as np
import pandas as pd

from skimage.feature import local_binary_pattern
from skimage.color import label2rgb
from skimage.transform import rotate
import matplotlib.pyplot as plt



image= cv2.imread("dataset/train/image/1.jpg")
#image = cv2.resize(image,(400,400))
#cv2.imwrite("resize_img.jpg",image)


image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
img2 = image.ravel()
df = pd.DataFrame()
df['Original Image'] = img2
# LBP Filters
num = 1  
num_g = 0
kernels = []
method = 'uniform'
print(image.shape)

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

from skimage.feature import graycomatrix, graycoprops
from skimage import data

x = image.shape[0]-6
y = image.shape[1]-6

PATCH_SIZE =440

max_pixel_value = np.amax(image)
desired_levels = 256 


if max_pixel_value >= desired_levels:
    image = (image * (desired_levels - 1) // max_pixel_value).astype(np.uint8)


num_glcm=0

k = 0


for i in range(image.shape[0]-x):
    print('%2.2f percent done'%(i/(image.shape[0]-PATCH_SIZE)*100))
    for j in range(image.shape[1]-y):
        print("ss")
        
        patch = image[i:i+x, j:j+y]
        for distance in range(1,2):
            for angle in range(2,4):

                glcm_label = 'GLCM' + str(num_glcm)
                glcm_label_diss = 'GLCM_diss' + str(num_glcm)
                glcm_label_corr = 'GLCM_corr' + str(num_glcm)
                
                glcm = graycomatrix(patch, [distance], [0, np.pi/1*angle, np.pi/1, 3*np.pi/1*angle], levels = desired_levels)
                diss = graycoprops(glcm, 'dissimilarity')[0, 0]
                corr = graycoprops(glcm, 'correlation')[0, 0]
             
                
                df[glcm_label_diss] = diss
                df[glcm_label_corr] = corr
               

                print(glcm_label, ': distance=', distance, ': angle=', np.pi/angle*2, np.pi/angle, 3*np.pi/angle*2)     
        num_glcm += 1
        
        for radius in range(1,5):   
            for n_point in range(1): 
                n_points =  radius * 8
                lbp_label = 'LBP' + str(num)  
                kernel_lbp = local_binary_pattern(patch, n_points, radius, method)   
                kernels.append(kernel_lbp)
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(image, cv2.CV_8UC3, kernel_lbp)
                #cv2.imshow("filter",fimg)
                #cv2.waitKey(0)
                filtered_img = fimg.reshape(-1)
                df[lbp_label] = filtered_img
                print(lbp_label, ': radius=', radius, ': n_points=', n_points, ': method=', method)
        num += 1  
                
       
        kernels_g = []
        for theta in range(2):
            theta = theta / 4. * np.pi
            for sigma in (1, 2):
                for lamda in np.arange(0, np.pi, np.pi / 4):
                    for gamma in (0.05, 0.2):
        #               print(theta, sigma, , lamda, frequency)
                        
                        gabor_label = 'Gabor' + str(num_g)
        #                    print(gabor_label)
                        ksize=9
                        kernel_g = cv2.getGaborKernel((i, j), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                        kernels_g.append(kernel_g)
                        
                        #Now filter image and add values to new column
                        fimg_g = cv2.filter2D(image, cv2.CV_8UC3, kernel_g)
                        
                        #cv2.imshow("filter",fimg_g)
                        #cv2.waitKey(0)
                        filtered_img_g = fimg_g.reshape(-1)
        
                        df[gabor_label] = filtered_img_g
                        
                        print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
        num_g += 1 
                        
        
labeled_img = cv2.imread('dataset/train/mask/1.jpg')
resized_img = cv2.resize(labeled_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)

labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
labeled_img1 = labeled_img.reshape(-1)
df['Labels'] = labeled_img1
#edges1_df = pd.DataFrame({'Labels': labeled_img1})
#df = pd.concat([df, edges1_df], axis=1)

###

Y = df["Labels"].values

X = df.drop(labels = ["Labels"], axis=1)   

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)


from sklearn.ensemble import RandomForestClassifier
# Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators = 100, random_state = 42) #n_estimators = 100 idi

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


feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp.head(20))


import pickle

#Save the trained model as pickle string to disk for future use
filename = "randomforest_model"
pickle.dump(model, open(filename, 'wb'))

#To test the model on future datasets
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(X)

segmented = result.reshape((image.shape))

from matplotlib import pyplot as plt
plt.subplot(221)
plt.imshow(image, cmap ='gray')
plt.subplot(222)
plt.imshow(labeled_img, cmap ='gray')
plt.subplot(224)
plt.imshow(segmented, cmap ='gray')
plt.imsave('RF_100_estim.jpg', segmented, cmap ='jet')

     