#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 19:53:55 2023

@author: esmanur
"""

import os
import cv2
import numpy as np

def crop_images(input_path, output_path, num_crops):
    image_files = [file for file in os.listdir(input_path) if not file.startswith('.')]

    for file in image_files:
        image = cv2.imread(os.path.join(input_path, file))
        height, width, _ = image.shape
        crop_size = min(height, width) // num_crops

        for i in range(num_crops):
            for j in range(num_crops):
                x = j * crop_size
                y = i * crop_size
                crop = image[y:y+crop_size, x:x+crop_size]

                output_file = os.path.splitext(file)[0] + f"_crop_{i}_{j}.jpg"
                output_file_path = os.path.join(output_path, output_file)
                cv2.imwrite(output_file_path, crop)

                
num_crops = 4  # Her bir resimden çıkarılacak kırpmaların sayısı

# Eğitim görüntülerini kırpma
input_train_path = 'membrane/train/image/'  # Eğitim görüntülerinin bulunduğu dizin
output_train_path = 'crops/train_image_crops/'  # Kırpılan görüntülerin kaydedileceği dizin
crop_images(input_train_path, output_train_path, num_crops)


# Test görüntüsünü kırpma
input_test_path = 'membrane/test/image/'  # Test görüntüsünün bulunduğu dizin
output_test_path = 'crops/test_image_crops/'  # Kırpılan görüntünün kaydedileceği dizin
crop_images(input_test_path, output_test_path, num_crops)

input_test_path = 'membrane/test/mask/'  # Test görüntüsünün bulunduğu dizin
output_test_path = 'crops/test_mask_crops/'  # Kırpılan görüntünün kaydedileceği dizin
crop_images(input_test_path, output_test_path, num_crops)

input_test_path = 'membrane/train/label/'  # Test görüntüsünün bulunduğu dizin
output_test_path = 'crops/train_mask_crops/'  # Kırpılan görüntünün kaydedileceği dizin
crop_images(input_test_path, output_test_path, num_crops)



