from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import BatchNormalization
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# %% Import dataset
dataset = pd.read_csv(
    'C:/Users/allen/Google Drive/School/UC/Semester 5/Pattern Recognition and Machine Learning/Semester Project/fer2013.csv')
# Processing image data to arrays
img_array = dataset.pixels.apply(lambda x: np.array(
    x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0)
# %%
# Processing labels to array
le = LabelEncoder()
img_labels = le.fit_transform(dataset.emotion)
img_labels = np_utils.to_categorical(img_labels)

# %%
# Splitting the dataset using the arrays which we just processed
# 20% of the total will be used for validation and the other 80% is used for training
X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels,
                                                      shuffle=True, stratify=img_labels,
                                                      test_size=0.2, random_state=1)

# %%
# Defining the layers of the CNN
# We have used the same layer configuration as Shawon

###############################
# Title: Facial Expression Detection (CNN)
# Author: Ashadullah Shawon
# Year: 2019
# Code Version: 3
# Availability: https://www.kaggle.com/shawon10/facial-expression-detection-cnn
###############################

model = keras.Sequential()
input_shape = (48, 48, 1)
model.add(Conv2D(64, (5, 5), input_shape=input_shape,
          activation='relu', padding='same'))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(7))
model.add(Activation('softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Show the layer summary
model.summary()

# %%
# Fitting the model (training)
# The model uses 20% of the data for the testing set and the rest for training
history = model.fit(img_array, img_labels, epochs=20,
                    batch_size=256, validation_split=0.2)
print(model.summary())
# %%

# Plot training and validation accuracy 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# %%
# Run evaluation on using validation set to get accuracy
accuracy = model.evaluate(X_valid, y_valid)

#%%
# Print the accuracy for each epoch to see the accuracy in more detail
print(history.history['accuracy'])
print(history.history['val_accuracy'])
#%%
# Plot loss curves
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss curves')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print(history.history['loss'])
print(history.history['val_loss'])

#%%
# Predict the examples in the validation set
y_pred = model.predict(X_valid)

#print the accuracy, precision, recall f1 score and classification report
print("Accuracy:",metrics.accuracy_score(y_valid, (y_pred > 0.5) ))
print("Precision:",metrics.precision_score(y_valid, (y_pred > 0.5) , average = 'weighted'))
print("Recall:",metrics.recall_score(y_valid, (y_pred > 0.5) , average = 'weighted'))
print("F1-score:",metrics.f1_score(y_valid, (y_pred > 0.5) , average = 'weighted'))
print(classification_report(y_valid, (y_pred > 0.5) ))
#%% 
# Print confusion Matrix

###############################
# Title: Facial Expression Detection (CNN)
# Author: Ashadullah Shawon
# Year: 2019
# Code Version: 3
# Availability: https://www.kaggle.com/shawon10/facial-expression-detection-cnn
###############################


cm = confusion_matrix(np.where(y_valid == 1)[1], y_pred.argmax(axis=1))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cm, index = [i for i in "0123456"],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (20,15))
sns.heatmap(df_cm, annot=True)