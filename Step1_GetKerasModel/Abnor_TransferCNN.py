## Written by Haoran Wei @UTD Nov 26,2018
## This code is used for CNN based Diabetic Retina Detection
## This moddel can be further used in an smartphone APP
import keras
import csv
import numpy
#import matplotlib
from sklearn.metrics import confusion_matrix


# Read label and image adress list from excel
with open('/Users/hxw170830/Desktop/DiabeticRetinaDetection/DataForRetina/STARE/Label0107_4.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    adress=[]
    label =[]
    y_actu=[]
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
            
        else:
            line_count += 1
            adress.append(['/Users/hxw170830/Desktop/DiabeticRetinaDetection/DataForRetina/STARE/JPEGimages/' +row[0]+ ".jpeg"])
            if int(row[1]) > 0:
                label.append([1, 0])
                y_actu.append(1)
            else:
                label.append([0 ,1])
                y_actu.append(0)
    print(f'Processed {line_count-1} lines.')
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
x_train = x[0:706] # include the head, not include the tail
x_test = x[706:717]
label_train = y[0:706]
label_test = y[706:717]
y_actu = y_actu[706:717]
del x,y, label, img
#mnist = tf.keras.datasets.mnist
#(mx_train, my_train),(mx_test, my_test) = mnist.load_data()
#mx_train, mx_test = mx_train / 255.0, mx_test / 255.0
#mx_train = mx_train.reshape(60000,28,28,1)
#mx_test = mx_test.reshape(10000,28,28,1)

 
# Transfer model
#model = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=(299, 299, 3), weights = "imagenet")
model = keras.applications.xception.Xception(include_top=False, input_shape=(299, 299, 3), weights = "imagenet")
# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:1]:
    layer.trainable = False
#Adding custom Layers 
x = model.get_layer('block14_sepconv2_act').output
x = keras.layers.GlobalMaxPooling2D()(x)
#x = keras.layers.Flatten()(x)
#x = tf.keras.layers.Dense(1024, activation="relu")(x)
#x = tf.keras.layers.Dropout(0.5)(x)
#x = tf.keras.layers.Dense(64, activation="relu")(x)
predictions = keras.layers.Dense(2, activation="softmax")(x)
# creating the final model 
model_final = keras.models.Model(inputs = model.input, outputs = predictions)
model_final.summary()
# compile the model 
#sgd = tf.keras.optimizers.SGD(lr=0.0008, decay=1e-6, momentum=0.9, nesterov=True)
model_final.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
# Train the model 
model_final.fit(x_train, label_train, epochs=1,batch_size=15,validation_split=0.1)


loss, accuracy = model_final.evaluate(x_test, label_test, batch_size=32)
classes = model_final.predict(x_test, batch_size=1)
y_pred = []
for j in range(len(classes)):
    if classes[j][0] >0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
ConfusionMatrix = confusion_matrix(y_actu, y_pred)
#keras.models.save_model(model_final,'myModel_TransferCNN.h5',overwrite=True,include_optimizer=True) 
#tf.keras.models.load_model(    filepath,    custom_objects=None,    compile=True)

#import coremltools
#coreml_model = coremltools.converters.keras.convert(model_final)
#coreml_model.save('myModel_TransferCNN.mlmodel')