
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('training_set',target_size=(64, 64), batch_size=32, class_mode='categorical')

test_set = test_datagen.flow_from_directory('test_set',target_size=(64, 64), batch_size=32, class_mode='categorical')





# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Poolin#g
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
#classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=2,activation='sigmoid'))

#classifier.add(Dense(units = 1, activation = 'sigmoid'))6666666666666

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[26]:


# Part 2 - Fitting the CNN to the images
#training_set = train_datagen.flow_from_directory(x_train,target_size = (64, 64),batch_size = 32,class_mode = 'binary')
#test_set = test_datagen.flow_from_directory(x_test,target_size = (64, 64),batch_size = 32,class_mode = 'binary')
#classifier.fit(training_set,test_set,epochs = 10, batch_size=100)
classifier.fit_generator(training_set,
steps_per_epoch = 8000,
epochs = 25,
validation_data = test_set,
validation_steps = 2000)

# In[27]:


y_pred = classifier.predict_classes(x_test)
x_pred = classifier.predict_classes(x_train)
print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test,y_pred))
print(f1_score(y_train,x_pred))


# In[25]:


#Model Save

classifier.save('my_model_BUSH_new.h5')  # creates a HDF5 file 'my_model.h5'
#del classifier  # deletes the existing model

# returns a compiled model
# identical to the previous one
classifier2 = load_model('my_model_BUSH.h5')
#print(classifier2)


# In[ ]:


print(classifier2.layers[0].get_config)
# evaluate loaded model on test data 
# Define X_test & Y_test data first
classifier2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = classifier2.evaluate(x_test, y_test, verbose=0)
print ("%s: %.2f%%" % (classifier2.metrics_names[0], score[1]*100))
print ("%s: %.2f%%" % (classifier2.metrics_names[1], score[1]*100))
y_pred = classifier2.predict_classes(x_test)
print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test,y_pred))


# In[19]:


#Williams

x_train_williams,x_test_williams,y_train_williams,y_test_williams=train_test_split(X_new1, y_williams,test_size=1./3 , random_state=int('5042'),shuffle = True,stratify=y_williams)


# In[ ]:


# Initialising the CNN
classifier_will = Sequential()
# Step 1 - Convolution
classifier_will.add(Conv2D(8, (3,3), input_shape = (64, 64, 1), activation = 'relu'))
# Step 2 - Pooling
classifier_will.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
#classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier_will.add(Flatten())
# Step 4 - Full connection
classifier_will.add(Dense(units=16, activation='relu'))
#classifier_will.add(Activation('tanh'))
classifier_will.add(Dense(1))
classifier_will.add(Activation('sigmoid'))
#classifier.add(Dense(units = 1, activation = 'sigmoid'))6666666666666

# Compiling the CNN
classifier_will.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


classifier_will.fit(x_train_williams,y_train_williams,epochs = 5, batch_size=100)


# In[ ]:


y_pred_williams = classifier_will.predict_classes(x_test_williams)
x_pred_williams= classifier_will.predict_classes(x_train_williams)
#print(y_pred_williams)


# In[ ]:


print(confusion_matrix(y_test_williams, y_pred_williams))
print(precision_score(y_test_williams, y_pred_williams))
print(recall_score(y_test_williams, y_pred_williams))
print(f1_score(y_test_williams,y_pred_williams))
print(f1_score(y_train_williams,x_pred_williams))


# In[6]:


#Plot
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

plot_model(classifier_will, to_file='model.pdf')
SVG(model_to_dot(classifier_will).create(prog='dot', format='svg'))


# In[17]:


#Model Save

classifier_will.save('my_model_williams_.h5')  # creates a HDF5 file 'my_model.h5'
#del classifier  # deletes the existing model

# returns a compiled model
# identical to the previous one
classifier2_will = load_model('my_model_williams_52.h5')
print(classifier2_will)


# In[22]:


#Model Load
print(classifier2_will.layers[0].get_config)
# evaluate loaded model on test data 
# Define X_test & Y_test data first
classifier2_will.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = classifier2_will.evaluate(x_test_williams, y_test_williams, verbose=0)

print ("%s: %.2f%%" % (classifier2_will.metrics_names[0], score[1]*100))
print ("%s: %.2f%%" % (classifier2_will.metrics_names[1], score[1]*100))
y_pred_williams = classifier2_will.predict_classes(x_test_williams)
x_pred_williams = classifier2_will.predict_classes(x_train_williams)
print(confusion_matrix(y_test_williams, y_pred_williams))
print(precision_score(y_test_williams, y_pred_williams))
print(recall_score(y_test_williams, y_pred_williams))
print(f1_score(y_test_williams, y_pred_williams))
print(f1_score(y_train_williams,x_pred_williams))


# In[23]:


#Plot loaded model
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

plot_model(classifier2_will, to_file='model.pdf')
SVG(model_to_dot(classifier2_will).create(prog='dot', format='svg'))

