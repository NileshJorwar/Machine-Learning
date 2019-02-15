# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

X=pd.read_csv('X.csv',sep=' ', header=None, dtype=float)
X=X.values

y=pd.read_csv('y_bush_vs_others.csv',header=None)
y_bush=y.values.ravel()
z=pd.read_csv('y_williams_vs_others.csv',header=None)
y_williams=z.values.ravel()

#Reshaping
X_new=[]
for i in range(len(X)):
    X_new.append(np.reshape(X[i],(64,64,1)))
X_new1=np.array(X_new)

plt.imshow(X_new1[0])
#Splitting
x_train,x_test,y_train,y_test=train_test_split(X_new1, y_bush,test_size=1./3 , random_state=int('5042'),shuffle = True,stratify=y)

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 1), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = None)


# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
#training_set = train_datagen.flow_from_directory(x_train,target_size = (64, 64),batch_size = 32,class_mode = 'binary')
#test_set = test_datagen.flow_from_directory(x_test,target_size = (64, 64),batch_size = 32,class_mode = 'binary')
classifier.fit_generator(x_train,steps_per_epoch = 8822,epochs = 25,validation_data = x_test,validation_steps = 4411)

# Part 3 - Making new predictions

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(x_test)

training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'