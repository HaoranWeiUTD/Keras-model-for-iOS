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
            adress.append([r"C:\Users\hxw170830\Desktop\DiabeticRetinaDetection\DataForRetina\FromKaggle\trainResizeTo224_224\Resize_"+row[0]+".jpeg"])
            if int(row[1]) > 0:
                label.append(1)
            else:
                label.append(0)
    print(f'Processed {line_count-1} lines.')
del  line_count,  row


# Read Image from the image adress list and show an example 
img = []
for i in range(len(adress)):
    images = tf.keras.preprocessing.image.load_img(adress[i][0], target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(images)
    x = numpy.expand_dims(x, axis=0)
    img.append(x/255.0)
    print('loading no.%s image' % i)
matplotlib.pyplot.imshow(img[35125][0])  
del adress, i, x, images
x = numpy.concatenate([x for x in img])
del img
# Divide Training and Testing Set
x_train = x[0:29999]
x_test = x[30000:35126]
label_train = label[0:29999]
label_test = label[30000:35126]
del x,label
#mnist = tf.keras.datasets.mnist
#(x_train, y_train),(x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0
#x_train = x_train.reshape(60000,28,28,1)
#x_test = x_test.reshape(10000,28,28,1)

 
# Define model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, kernel_size=3, activation=tf.nn.relu,padding="same",input_shape=(224,224,3)),
  tf.keras.layers.Conv2D(32, kernel_size=3, activation=tf.nn.relu,padding="same"),
  tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)), 
  tf.keras.layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu,padding="same"),
  tf.keras.layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu,padding="same"),
  tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),   
  tf.keras.layers.Conv2D(128, kernel_size=3, activation=tf.nn.relu,padding="same"),
  tf.keras.layers.Conv2D(128, kernel_size=3, activation=tf.nn.relu,padding="same"),
  tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),   
  tf.keras.layers.Conv2D(256, kernel_size=3, activation=tf.nn.relu,padding="same"),
  tf.keras.layers.Conv2D(256, kernel_size=3, activation=tf.nn.relu,padding="same"),
  tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)), 
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(4096, activation=tf.nn.relu),
  tf.keras.layers.Dense(1024, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5,batch_size=32)
loss, accuracy = model.evaluate(x_test, label_test, batch_size=1)
#classes = model.predict(x_test, batch_size=1)
tf.keras.models.save_model(model,'myModel_PaperCNN.h5',overwrite=True,include_optimizer=True) 
#tf.keras.models.load_model(    filepath,    custom_objects=None,    compile=True)