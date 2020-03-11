## Written by Haoran Wei @UTD Nov 26,2018
## This code is used for CNN based Diabetic Retina Detection
## This moddel can be further used in an smartphone APP
import tensorflow as tf
import csv
import numpy
import matplotlib

# Read label and image adress list from excel
with open(r'C:\Users\hxw170830\Desktop\DiabeticRetinaDetection\DataForRetina\FromKaggle\trainLabels\trainLabels.csv') as csv_file:
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
            adress.append(["C:\\Users\\hxw170830\\Desktop\\DiabeticRetinaDetection\\DataForRetina\\FromKaggle\\train\\" +row[0]+ ".jpeg"])
            if int(row[1]) > 0:
                label.append([1, 0])
            else:
                label.append([0 ,1])
    print(f'Processed {line_count-1} lines.')
del  line_count,  row
y = numpy.array(label)

# Read Image from the image adress list and show an example 
img = []
for i in range(len(adress)):
    images = tf.keras.preprocessing.image.load_img(adress[i][0], target_size=(299, 299))
    x = tf.keras.preprocessing.image.img_to_array(images)
    x = numpy.expand_dims(x, axis=0)
    img.append(x/255.0)
    print('loading no.%s image' % i)
#matplotlib.pyplot.imshow(img[35125][0])  
del adress, i, x, images
x = numpy.concatenate([x for x in img])
# Divide Training and Testing Set
x_train = x[0:30000] # include the head, not include the tail
x_test = x[30000:35126]
label_train = y[0:30000]
label_test = y[30000:35126]
del x,y, label, img
#mnist = tf.keras.datasets.mnist
#(mx_train, my_train),(mx_test, my_test) = mnist.load_data()
#mx_train, mx_test = mx_train / 255.0, mx_test / 255.0
#mx_train = mx_train.reshape(60000,28,28,1)
#mx_test = mx_test.reshape(10000,28,28,1)

 
# Transfer model
#model = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=(299, 299, 3), weights = "imagenet")
model = tf.keras.applications.xception.Xception(include_top=True, weights = "imagenet")
# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:1]:
    layer.trainable = False
#Adding custom Layers 
x = model.get_layer('avg_pool').output
x = tf.keras.layers.Flatten()(x)
#x = tf.keras.layers.Dense(1024, activation="relu")(x)
#x = tf.keras.layers.Dropout(0.5)(x)
#x = tf.keras.layers.Dense(64, activation="relu")(x)
predictions = tf.keras.layers.Dense(2, activation="softmax")(x)
# creating the final model 
model_final = tf.keras.models.Model(inputs = model.input, outputs = predictions)
model_final.summary()
# compile the model 
#sgd = tf.keras.optimizers.SGD(lr=0.0008, decay=1e-6, momentum=0.9, nesterov=True)
model_final.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
# Train the model 
model_final.fit(x_train, label_train, epochs=20,batch_size=16,validation_split=0.1)


loss, accuracy = model_final.evaluate(x_test, label_test, batch_size=32)
classes = model_final.predict(x_test, batch_size=1)
tf.keras.models.save_model(model_final,'myModel_TransferCNN.h5',overwrite=True,include_optimizer=True) 
#tf.keras.models.load_model(    filepath,    custom_objects=None,    compile=True)

#import coremltools
#coreml_model = coremltools.converters.keras.convert(model_final)
#coreml_model.save('myModel_TransferCNN.mlmodel')