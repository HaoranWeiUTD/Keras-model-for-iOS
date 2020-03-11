# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:57:41 2019

@author: hxw170830
"""

## Written by Haoran Wei @UTD Nov 26,2018
## This code is used for CNN based Diabetic Retina Detection
## This moddel can be further used in an smartphone APP
import keras
import csv
import numpy
from sklearn.metrics import confusion_matrix
import_model = keras.models.load_model( "C:/Users/hxw170830/Desktop/DiabeticRetinaDetection/KerasCode/myModel_TransferCNN_DataBalanced4-6.h5",    custom_objects=None,    compile=True)


# Read label and image adress list from excel
with open(r'C:\Users\hxw170830\Desktop\DiabeticRetinaDetection\DataForRetina\FromKaggle\trainLabels\trainLabels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    adress=[]
    label =[]
    y_actu=[]
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
            
        elif line_count < 30000:
            line_count += 1
            #adress.append(["C:\\Users\\hxw170830\\Desktop\\DiabeticRetinaDetection\\DataForRetina\\FromKaggle\\train\\" +row[0]+ ".jpeg"])
            if int(row[1]) > 2:
                label.append([1, 0])
                y_actu.append(1)
                adress.append(["C:\\Users\\hxw170830\\Desktop\\DiabeticRetinaDetection\\DataForRetina\\FromKaggle\\train\\" +row[0]+ ".jpeg"])
            elif int(row[1]) == 0:
                label.append([0 ,1])
                y_actu.append(0)
                adress.append(["C:\\Users\\hxw170830\\Desktop\\DiabeticRetinaDetection\\DataForRetina\\FromKaggle\\train\\" +row[0]+ ".jpeg"])    
        else:
            line_count += 1            
            if int(row[1]) > 2:
                label.append([1, 0])
                y_actu.append(1)
                adress.append(["C:\\Users\\hxw170830\\Desktop\\DiabeticRetinaDetection\\DataForRetina\\FromKaggle\\train\\" +row[0]+ ".jpeg"])
            elif int(row[1]) == 0:
                label.append([0 ,1])
                y_actu.append(0)
                adress.append(["C:\\Users\\hxw170830\\Desktop\\DiabeticRetinaDetection\\DataForRetina\\FromKaggle\\train\\" +row[0]+ ".jpeg"])
            
del  line_count,  row
y = numpy.array(label)

# Read Image from the image adress list and show an example 
img = []
for i in range(len(adress)):
    images = keras.preprocessing.image.load_img(adress[i][0], target_size=(299, 299))
    x = keras.preprocessing.image.img_to_array(images)
    x = numpy.expand_dims(x, axis=0)
    img.append(x/255.0)
    print('loading no.%s image' % i)
#matplotlib.pyplot.imshow(img[35125][0])  
del adress, i, x, images
x = numpy.concatenate([x for x in img])
# Divide Training and Testing Set
x_train = x[0:23354] # include the head, not include the tail
x_test = x[23354:27391]
label_train = y[0:23354]
label_test = y[23354:27391]
y_actu = y_actu[23354:27391]
del x,y, label, img

# Transfer model
import_model.fit(x_train, label_train, epochs=1,batch_size=12,validation_split=0.1)

loss, accuracy = import_model.evaluate(x_test, label_test, batch_size=32)
classes = import_model.predict(x_test, batch_size=32)
y_pred = []
for j in range(len(classes)):
    if classes[j][0] >0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
ConfusionMatrix = confusion_matrix(y_actu, y_pred)
keras.models.save_model(import_model,'myModel_TransferCNN_severe.h5',overwrite=True,include_optimizer=True) 
#import_model = keras.models.load_model( "C:/Users/hxw170830/Desktop/DiabeticRetinaDetection/KerasCode/myModel_TransferCNN1207_83.h5",    custom_objects=None,    compile=True)

#import coremltools
#coreml_model = coremltools.converters.keras.convert(model_final)
#coreml_model.save('myModel_TransferCNN.mlmodel')