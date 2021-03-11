#!/usr/bin/env python
# coding: utf-8

# ### 1. Key Facial Points Detection
# - The dataset consists of x and y coordinates of 15 facial key points.
# - Input images are 96 x 96 pixels
# - Images are greyscaled or single channel

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import os
import PIL
import pickle
from PIL import *
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121 # 2017 architecture
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


# Load data
keyfacial_df = pd.read_csv('./1.2 Emotion AI Dataset/Emotion AI Dataset/data.csv')
# image column contains pixel values(0to255) for each row


# In[3]:


keyfacial_df #2140 rows × 31 columns


# In[4]:


keyfacial_df.info() # no missing values!!


# In[5]:


# check if null values exist
keyfacial_df.isnull().sum()


# In[6]:


print(keyfacial_df['Image'].shape, keyfacial_df['mouth_left_corner_x'].shape)


# In[7]:


# our image column contains string values, saperated with space
# we will convert this into numpy array using np.frontstring and
# convert the obtained 1D array into 2D array of shape (96,96)
keyfacial_df['Image'] = keyfacial_df['Image'].apply(lambda x : np.fromstring(x, dtype = int, sep = ' ').reshape(96,96))


# In[8]:


keyfacial_df['Image'][0].shape


# In[9]:


# print(keyfacial_df['mouth_center_top_lip_x'].min(),'\n',
#       keyfacial_df['mouth_center_top_lip_x'].mean(),'\n',
#       keyfacial_df['mouth_center_top_lip_x'].median(),'\n',
#       keyfacial_df['mouth_center_top_lip_x'].max())
keyfacial_df.describe()


# ### Visualization

# In[10]:


# Plot any random image from dataset along with facial keypoints
# image data is obtained from keyfacial_df['Image'] 
i = np.random.randint(1, len(keyfacial_df))
plt.imshow(keyfacial_df['Image'][i], cmap ='gray')
for j in range(1, 31, 2): # no. of cols = 30, so from 1 to 30 with jump of 2 plot all X cords and Y cords
    plt.plot(keyfacial_df.loc[i][j-1], keyfacial_df.loc[i][j], 'ro') # [j-1] mean x [j] mean y


# In[11]:


# display more images
# this is known as sanity check  
fig = plt.figure(figsize = (20,20))
for i in range(16):
    ax = fig.add_subplot(4,4,i+1)
    image = plt.imshow(keyfacial_df['Image'][i], cmap = 'gray')
    for j in range(1,31,2):
        plt.plot(keyfacial_df.loc[i][j-1], keyfacial_df.loc[i][j],'yo')


# ### Image Augmentation
# - As we know deep learning is hungary of datas
# - Image Augmentation is used to generate more datas using input data
# - It may flip image horizontally, vertically, zoom it, zoom out etc..
# 

# In[12]:


import copy 
keyfacial_df_copy = copy.copy(keyfacial_df)


# In[13]:


# obtain columns in the dataframe
columns = keyfacial_df_copy.columns[:-1] #ignoring Image column
len(columns)


# In[14]:


# horizontal flip  - flip the images along y-axis 
keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x : np.flip(x , axis = 1))

# on flipping X coords with change but Y won't, we will subtract out initial x coord values from the width of the image
for i in range(len(columns)):
    if i%2 == 0: # selecting even cols that are x cols
        keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(lambda x : 96. - float(x))


# In[15]:


# original Image
plt.imshow(keyfacial_df['Image'][0], cmap = 'gray')
for j in range(1, 31, 2):
    plt.plot(keyfacial_df.loc[0][j-1], keyfacial_df.loc[0][j], 'yo')


# In[16]:


# horizonatally flipped image
plt.imshow(keyfacial_df_copy['Image'][0], cmap = 'gray')
for j in range(1, 31, 2):
    plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'yo')


# In[17]:


# concatenate original and flipped image 
augmented_df = np.concatenate((keyfacial_df, keyfacial_df_copy))


# In[18]:


augmented_df.shape


# In[19]:


# augmentation using change in brightness
# multiply pixel values by random values between 1.5 to 2 increase the brightness of the image
# we clip the value btw 0 and 255

import random

keyfacial_df_copy = copy.copy(keyfacial_df)
keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x:np.clip(random.uniform(1.5, 2)* x, 0.0, 255.0))
augmented_df = np.concatenate((augmented_df, keyfacial_df_copy))
augmented_df.shape


# In[20]:


plt.imshow(keyfacial_df_copy['Image'][0], cmap = 'gray')
for j in range(1, 31, 2):
    plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'bo')


# In[21]:


keyfacial_df_copy = copy.copy(keyfacial_df)


# In[22]:


# # vertically flip  - flip the images along x-axis 
# keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x : np.flip(x , axis = 0))

# # on flipping X coords with change but Y won't, we will subtract out initial x coord values from the width of the image
# for i in range(len(columns)):
#     if i%2 != 0: # selecting even cols that are x cols
#         keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(lambda x : 96. - float(x))


# In[23]:


# # vertically flipped image
# plt.imshow(keyfacial_df_copy['Image'][0], cmap = 'gray')
# for j in range(1, 31, 2):
#     plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'yo')


# In[24]:


augmented_df = np.concatenate((augmented_df, keyfacial_df_copy))
augmented_df.shape


# ### Data Normalization and Data Prepration

# In[25]:


# Obtain the value of images which is present in the 31st columns
# 30 is the last column as indexing is from 0
img = augmented_df[:,30]
len(img)


# In[26]:


# Normalize The image
img = img / 255.

#create an empty array of shape (x, 96, 96, 1) to feed the model
X = np.empty((len(img), 96, 96,1))

# iterate through the image list and add image values to the empty array
# after expanding its dimension from (96,96) to (96,96,1)
for i in range(len(img)):
    X[i,] = np.expand_dims(img[i], axis = 2)

# Convert the array type to float 32
X = np.asarray(X).astype(np.float32)
X.shape


# In[27]:


# print(X[0])
print(len(X[0]))


# In[28]:


# Values of X n Y coords which are to used as target
y = augmented_df[:, :30]
y = np.asarray(y).astype(np.float32)
y.shape


# In[29]:


y[0]


# In[30]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[31]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[32]:


# Dense ANN mean all the neuron in a layer is fully connected to all neurons in the subsequent layer


# ### RESNET (RESIDUAL NETWORK)
# - As CNNs grows deeper, vanishing gradient tend to occurs which negatively impact network performance.
# - Vanishing Gradient problem occurs when the gradient is back-propagated to earlier layers which results in a very small gradient.
# - Residual Neural Network includes 'skip connection' feature which enables training of 152 layers without vanishing gradient issues.
# - RESNET works by adding 'identity mappings' on top of CNN.
# - ImageNet contains 11M images and 11000 categories.
# - ImageNet is used to train ResNet deep Network.

# ### Building ResNet to Detect key Facial Points

# In[33]:


# RESBLOCK :=> input -> Convolution Block -> Identity Block -> Identity Block -> output
def res_block(X, filters,stage):
    # Convolution Block
    X_copy = X

    f1, f2, f3 = filters
    
    # main path
    X = Conv2D(f1,(1,1), strides = (1,1), name = 'res_'+str(stage)+'_conv_a', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = MaxPool2D((2,2))(X)
    X = BatchNormalization(axis = 3, name = 'bn_'+ str(stage) + '_conv_a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f2,(3,3), strides = (1,1), padding= 'same', name = 'res_' + str(stage) + '_conv_b', kernel_initializer = glorot_uniform(seed = 0))(X)

    X = BatchNormalization(axis = 3, name = 'bn_'+ str(stage) + '_conv_b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f3,(1,1),strides = (1,1), name ='res_' + str(stage) + '_conv_c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_'+ str(stage) + '_conv_c')(X)
    
    
    # Short path
    
    X_copy = Conv2D(f3,(1,1), strides =(1,1), name = 'res_'+str(stage) +'_conv_copy', kernel_initializer = glorot_uniform(seed = 0))(X_copy)
    X_copy = MaxPool2D((2,2))(X_copy)
    X_copy = BatchNormalization(axis = 3, name='bn_'+str(stage)+'_conv_copy')(X_copy)
    
    # ADD
    X = Add()([X,X_copy])
    X = Activation('relu')(X)
    
    # Identity BLock 1
    X_copy = X 
    
    # main path
    # main path
    X = Conv2D(f1,(1,1), strides = (1,1), name = 'res_'+str(stage)+'_identity_1_a', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_'+ str(stage) + '_identity_1_a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f2,(3,3), strides = (1,1), padding= 'same', name = 'res_' + str(stage) + '_identity_1_b', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_'+ str(stage) + '_identity_1_b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f3, (1,1),strides = (1,1), name ='res_' + str(stage) + '_identity_1_c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_'+ str(stage) + '_identity_1_c')(X)
    
    # ADD
    X = Add()([X,X_copy])
    X = Activation('relu')(X)
    
    # identity block 2
    X_copy = X 
    
    # main path
    X = Conv2D(f1,(1,1), strides = (1,1), name = 'res_'+str(stage)+'_identity_2_a', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_'+ str(stage) + '_identity_2_a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f2,(3,3), strides = (1,1), padding= 'same', name = 'res_' + str(stage) + '_identity_2_b', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_'+ str(stage) + '_identity_2_b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f3, (1,1),strides = (1,1), name ='res_' + str(stage) + '_identity_2_c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_'+ str(stage) + '_identity_2_c')(X)
    # ADD
    X = Add()([X,X_copy])
    X = Activation('relu')(X)
    
    return X


# In[34]:


input_shape = (96, 96, 1)

# Input tensor shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3,3))(X_input)

# 1 - stage
X = Conv2D(64, (7,7), strides= (2,2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3,3), strides= (2,2))(X)

# 2 - stage
X = res_block(X, (64,64,256), stage= 2)

# 3 - stage
X = res_block(X, (128,128,512), stage= 3)


# Average Pooling
X = AveragePooling2D((2,2), name = 'Average_Pooling')(X)

# Final layer
X = Flatten()(X)
X = Dense(4096, activation = 'relu')(X)
X = Dropout(0.2)(X)
X = Dense(2048, activation = 'relu')(X)
X = Dropout(0.1)(X)
X = Dense(30, activation = 'relu')(X)


model_1_facialKeyPoints = Model( inputs= X_input, outputs = X)
model_1_facialKeyPoints.summary()


# In[35]:


adam = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
model_1_facialKeyPoints.compile(loss = "mean_squared_error", optimizer = adam , metrics = ['accuracy'])


# In[36]:


# save the best model with least validation loss
checkpointer = ModelCheckpoint(filepath = "FacialKeyPoints_weights.hdf5", verbose = 1, save_best_only = True)


# In[37]:


history =  model_1_facialKeyPoints.fit(X_train, y_train, batch_size = 10, epochs = 2, validation_split = 0.05, callbacks=[checkpointer])


# In[38]:


# save model architecture to a json file
# storing features like no. of layers, no. of maxpooling etc of model

model_json = model_1_facialKeyPoints.to_json()
with open('FacialKeyPoitns-model.json','w') as json_file:
    json_file.write(model_json)


# In[39]:


with open('C:/Users/dell/Desktop/Machine Learning/Project Emotion AI/1.2 Emotion AI Dataset/Emotion AI Dataset/detection.json', 'r') as json_file:
    json_savedModel = json_file.read()
    
# load the model architecture
model1_facialKeyPoints = tf.keras.models.model_from_json(json_savedModel)
model1_facialKeyPoints.load_weights('C:/Users/dell/Desktop/Machine Learning/Project Emotion AI/1.2 Emotion AI Dataset/Emotion AI Dataset/weights_keypoint.hdf5')
adam = tf.keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
model1_facialKeyPoints.compile(loss = "mean_squared_error", optimizer = adam , metrics = ['accuracy'])


# In[40]:


result = model_1_facialKeyPoints.evaluate(X_test, y_test)
print("Accuracy : {}".format(result[1]))


# In[41]:


history.history.keys()


# In[42]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('Epochs')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()


# ### FACIAL EXPRESSION DETECTION

# **Categories**
# - 0 - Angry
# - 1 - Disgust
# - 2 - Sad
# - 3 - Happy
# - 4 - Suprise

# In[43]:


# read the csv files for the facial expression data
facialexpression_df = pd.read_csv('C:/Users/dell/Desktop/Machine Learning/Project Emotion AI/1.2 Emotion AI Dataset/Emotion AI Dataset/icml_face_data.csv')


# In[44]:


facialexpression_df.head()


# In[45]:


facialexpression_df[' pixels'][0]


# In[46]:


# fucntipn to convert pixel values in string format to array format
def string2array(x):
    return np.array(x.split(' ')).reshape(48,48,1).astype('float32')


# In[47]:


# resize images from (48,48) to (96,96)
def resize(x):
    img = x.reshape(48,48)
    return cv2.resize(img, dsize=(96,96), interpolation = cv2.INTER_CUBIC)


# In[48]:


facialexpression_df[' pixels'] = facialexpression_df[' pixels'].apply(lambda x : string2array(x))


# In[49]:


facialexpression_df[' pixels'] =  facialexpression_df[' pixels'].apply(lambda x : resize(x))


# In[50]:


facialexpression_df.head()


# In[51]:


facialexpression_df.shape


# In[52]:


facialexpression_df.isnull().sum()


# In[53]:


label_to_text = {0: 'anger', 1 : 'disgust', 2 : 'sad', 3 : 'happiness', 4: 'surprise'}


# In[54]:


plt.axis('off')
plt.imshow(facialexpression_df[' pixels'][0],cmap = 'gray')
plt.show()


# In[55]:


#visualize images and plot labels
emotions = [0,1,2,3,4]
for i in emotions:
    data = facialexpression_df[facialexpression_df['emotion']==i][:1]
    plt.axis('off')
    img = data[' pixels'].item()
    img = img.reshape(96, 96)
    
    plt.figure()
    plt.title(label_to_text[i])
    plt.imshow(img, cmap = 'gray')


# In[56]:


facialexpression_df.emotion.value_counts().index


# In[57]:


facialexpression_df.emotion.value_counts()


# In[58]:


#facialexpression_df.emotion.value_counts().index
#facialexpression_df.emotion.value_counts()
plt.figure(figsize = (10,10))
sns.barplot(x = facialexpression_df.emotion.value_counts().index,y = facialexpression_df.emotion.value_counts() )


# In[59]:


# we will do image augmentation on class 1 since it has very small features


# In[60]:


# split ddataframe in to feauters and labels
from keras.utils import to_categorical

X = facialexpression_df[' pixels']
y = to_categorical(facialexpression_df['emotion'])

X = np.stack(X, axis = 0)
X = X.reshape(24568,96,96,1)

print(X.shape, y.shape)


# In[61]:


X


# In[62]:



X = np.stack(X, axis = 0)
X = X.reshape(24568, 96, 96, 1)

print(X.shape, y.shape)


# In[63]:


X[0]


# In[64]:


y


# In[65]:


from sklearn.model_selection import train_test_split
X_train, X_Test, y_train, y_Test = train_test_split(X, y, test_size = 0.1, shuffle = True)
X_val, X_Test, y_val, y_Test = train_test_split(X_Test, y_Test, test_size = 0.5, shuffle = True)


# In[66]:


X_val.shape, y_val.shape


# In[67]:


X_train = X_train/255
X_val   = X_val /255
X_Test  = X_Test/255


# In[68]:


train_datagen = ImageDataGenerator(
rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    fill_mode = "nearest")


# ### Building Facial Expression Classifier

# In[69]:


input_shape = (96, 96, 1)

# Input tensor shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3, 3))(X_input)

# 1 - stage
X = Conv2D(64, (7, 7), strides= (2, 2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides= (2, 2))(X)

# 2 - stage
X = res_block(X, [64, 64, 256], stage= 2)

# 3 - stage
X = res_block(X, [128, 128, 512], stage= 3)

# 4 - stage
# X = res_block(X, filter= [256, 256, 1024], stage= 4)

# Average Pooling
X = AveragePooling2D((4, 4), name = 'Averagea_Pooling')(X)

# Final layer
X = Flatten()(X)
X = Dense(5, activation = 'softmax', name = 'Dense_final', kernel_initializer= glorot_uniform(seed=0))(X)

model_2_emotion = Model( inputs= X_input, outputs = X, name = 'Resnet18')

model_2_emotion.summary()


# In[70]:


model_2_emotion.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# In[71]:



# using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath = "FacialExpression_weights.hdf5", verbose = 1, save_best_only=True)


# In[72]:


history = model_2_emotion.fit(train_datagen.flow(X_train, y_train, batch_size=64),
	validation_data= (X_val, y_val), steps_per_epoch=len(X_train) // 64,
	epochs= 2, callbacks=[checkpointer, earlystopping])


# In[73]:


model_json = model_2_emotion.to_json()
with open("FacialExpression-model.json","w") as json_file:
  json_file.write(model_json)


# Confusion Matrix, Accuracy, Precision, And Recall these are KPIs (Key Performance Indicator)

# In[74]:


with open('C:/Users/dell/Desktop/Machine Learning/Project Emotion AI/1.2 Emotion AI Dataset/Emotion AI Dataset/emotion.json', 'r') as json_file:
    json_savedModel= json_file.read()
    
# load the model architecture 
model_2_emotion = tf.keras.models.model_from_json(json_savedModel)
model_2_emotion.load_weights('C:/Users/dell/Desktop/Machine Learning/Project Emotion AI/1.2 Emotion AI Dataset/Emotion AI Dataset/weights_emotions.hdf5')
model_2_emotion.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# In[75]:


score = model_2_emotion.evaluate(X_Test, y_Test)
print('Test Accuracy: {}'.format(score[1]))


# In[76]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[77]:


epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'rx', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


# In[78]:


plt.plot(epochs, loss, 'bx', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()


# In[79]:


predicted_classes = np.argmax(model_2_emotion.predict(X_Test), axis=-1)
y_true = np.argmax(y_Test, axis=-1)


# In[80]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True, cbar = False)


# In[81]:


L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize = (24, 24))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i].reshape(96,96), cmap = 'gray')
    axes[i].set_title('Prediction = {}\n True = {}'.format(label_to_text[predicted_classes[i]], label_to_text[y_true[i]]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)   


# In[82]:


from sklearn.metrics import classification_report
print(classification_report(y_true, predicted_classes))


# #### COMBINE BOTH FACIAL EXPRESSION AND KEY POINTS DETECTION MODELS

# In[83]:


def predict(X_test):
    # making predictions from the keypoint model
    df_predict = model_1_facialKeyPoints.predict(X_test)
    
    # making predictions from emotion model
    df_emotion = np.argmax(model_2_emotion.predict(X_test), axis = 1)  #max prediction out of five o/ps
    
    # Reshapining array from (856,) to (856, 1)
    df_emotion = np.expand_dims(df_emotion, axis =1)
    
    # Converting the predictions into a dataframe
    df_predict = pd.DataFrame(df_predict, columns = columns)
    
    # adding emotions into the predicted dataframe
    df_predict['emotion'] = df_emotion
    
    return df_predict


# In[84]:


df_predict = predict(X_test)


# In[85]:


df_predict.head()


# In[86]:


fig, axes = plt.subplots(5, 5, figsize = (24, 24))
axes = axes.ravel()
for i in range(25):

    axes[i].imshow(X_test[i].squeeze(),cmap='gray')
    axes[i].set_title('Prediction = {}\n True = {}'.format(label_to_text[df_predict['emotion'][i]], label_to_text[y_true[i]]))
    axes[i].axis('off')
    for j in range(1,31,2):
            axes[i].plot(df_predict.loc[i][j-1], df_predict.loc[i][j], 'yo')
            


# In[ ]:




