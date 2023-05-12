#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:53:47 2023

@author: esmanur
"""
import cv2
import numpy as np
import pandas as pd

from skimage.feature import local_binary_pattern
import pickle
 
image= cv2.imread("dataset/img/img-1.png")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

img2 = image.ravel()
df = pd.DataFrame()
df['Original Image'] = img2

# LBP Filters
num = 1  
kernels = []
method = 'uniform'

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
for theta in range(2):
    theta = theta / 4. * np.pi
    for sigma in (1, 2):
        for lamda in np.arange(0, np.pi, np.pi / 4):
            for gamma in (0.05, 0.2):
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





import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from skimage import data


PATCH_SIZE = 21

# open the camera image
num_glcm = 0
grass_locations = [(780, 490), (842, 483), (744, 592), (655, 655)]
grass_patches = []
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])
    
    
for distance in range(1,2):
    for angle in range(2,4):
        glcm_label = 'GLCM' + str(num_glcm)
        # select some patches from grassy areas of the image
        
        for patch in (grass_patches):
            glcm = graycomatrix(patch, [distance], [0, np.pi/angle*2, np.pi/angle, 3*np.pi/angle*2])
                
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
df['Variance s3'] = variance_img1  #Add column to original dataframe
    
        
  
labeled_img = cv2.imread('dataset/inf/mask-1.png')
resized_img = cv2.resize(labeled_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)

labeled_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
labeled_img1 = labeled_img.reshape(-1)
df['Labels'] = labeled_img1        
        
Y = df["Labels"].values
columns=["Gaussian s7","Gabor32","Gabor7","Gabor8","Gabor6","Gabor24","Gabor28","Gabor22","Gabor23","Prewitt"]

X = df.loc[:,columns]
#Define the independent variables
X = df.drop(labels = ["Labels"], axis=1) 

"""
#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=100, random_state=0)


# Train the model on training data
model.fit(X_train, y_train)


prediction_test_train = model.predict(X_train)

#Test prediction on testing data. 
prediction_test = model.predict(X_test)

#Let us check the accuracy on test data
from sklearn import metrics
#Print the prediction accuracy

#First check the accuracy on training data. This will be higher than test data prediction accuracy.
print ("Accuracy on training data = ", metrics.accuracy_score(y_train, prediction_test_train))
#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))


feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)


import pickle

#Save the trained model as pickle string to disk for future use
filename = "sandstone_model"
pickle.dump(model, open(filename, 'wb'))
"""

import pickle
import glob
filename = "sandstone_model"
#To test the model on future datasets
loaded_model = pickle.load(open(filename, 'rb'))

i=0
#Call the feature extraction function.
result = loaded_model.predict(X)
segmented = result.reshape((image.shape))


#plt.imsave('images/'+ name[1], segmented, cmap ='jet')
from matplotlib import pyplot as plt
plt.subplot(211)
plt.imshow(image)
plt.subplot(212)
plt.imshow(segmented, cmap ='jet')
plt.imsave('segmentedrf'+str(i+1)+'.png', segmented, cmap ='jet')
 
    
    

