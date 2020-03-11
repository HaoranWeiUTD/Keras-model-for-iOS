# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:34:54 2018

@author: hxw170830
"""
import csv
import tensorflow as tf
import numpy as np
import matplotlib

with open(r'C:\Users\hxw170830\Desktop\trainLabels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    adress=[]
    label =[]
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
            
        else:
            line_count += 1
            adress.append([r"C:\Users\hxw170830\Desktop\DiabeticRetinaDetection\DataForRetina\FromKaggle\trainResizeTo224_224\Resize_"+row[0]+".jpeg"])
            if int(row[1]) > 0:
                label.append(1)
            else:
                label.append(0)
    print(f'Processed {line_count-1} lines.')


img = []
# 把图片读取出来放到列表中
for i in range(len(adress)):
    images = tf.keras.preprocessing.image.load_img(adress[i][0], target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(images)
    x = np.expand_dims(x, axis=0)
    img.append(x/255.0)
    print('loading no.%s image' % i)
    
matplotlib.pyplot.imshow(img[35125][0])    
    
    
    

