import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
import cv2
import os

# %% prepretation of the data

labels = ["PNEUMONIA", "NORMAL"]
image_size = 150

def get_data(data_direction):
    data = []
    for label in labels:
        path = os.path.join(data_direction, label)
        class_number = labels.index(label)
        
        for image in os.listdir(path):
            try:
                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(image_array, (image_size, image_size))
                data.append([resized_array, class_number])
            except Exception as e:
                print(e)
                
    return np.array(data, dtype="object")

train = get_data("dataset/report/chest_xray/train")
test = get_data("dataset/report/chest_xray/test")
val = get_data("dataset/report/chest_xray/val")

# %% number of images in train data

tags_train = []
for each in train:
    if(each[1] == 0):
        tags_train.append("Train-Pneumonia")
    else:
        tags_train.append("Train-Normal")
sns.set_style("darkgrid")
sns.countplot(x=tags_train) 
plt.show()  

# %% number of images in test data

tags_test = []
for each in test:
    if(each[1] == 0):
        tags_test.append("Test-Pneumonia")
    else:
        tags_test.append("Test-Normal")
sns.set_style("darkgrid")
sns.countplot(x=tags_test) 
plt.show() 

# %% number of images in validation data

tags_validation = []
for each in val:
    if(each[1] == 0):
        tags_validation.append("Validation-Pneumonia")
    else:
        tags_validation.append("Validation-Normal")
sns.set_style("darkgrid")
sns.countplot(x=tags_validation)
plt.show()  

# %% how do normal and penuomnia look

plt.figure(figsize = (5,5))
plt.imshow(train[0][0], cmap="gray")
plt.title(labels[train[0][1]])
plt.show()

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0], cmap="gray")
plt.title(labels[train[-1][1]])
plt.show()

# %% seperation of the images as the feature and the label of the feature in train, test, and validation sets

x_train = []
y_train = []

x_test = []
y_test = []

x_val = []
y_val = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
    
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)
    
# %% normalize data

x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

# %% resize data 

x_train = x_train.reshape(-1, image_size, image_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, image_size, image_size, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, image_size, image_size, 1)
y_test = np.array(y_test)

# %% data augmentation to prevent overfitting and handling the imbalance
"""
For the data augmentation:
Randomly rotate some training images by 30 degrees
Randomly Zoom by 20% some training images
Randomly shift images horizontally by 10% of the width
Randomly shift images vertically by 10% of the height
Randomly flip images horizontally. Once our model is ready, we fit the training dataset.
"""

data_generator = ImageDataGenerator(
    featurewise_center=False, 
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)

data_generator.fit(x_train)

# %% training the model

model = Sequential()
model.add(Conv2D(32, (3,3), strides=1, padding="same", activation="relu", input_shape=(150,150,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding="same"))
model.add(Conv2D(64, (3,3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding="same"))
model.add(Conv2D(64, (3,3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding="same"))
model.add(Conv2D(128, (3,3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding="same"))
model.add(Conv2D(256, (3,3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

# %% 

learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience=2, verbose=1, factor=0.3, min_lr=0.000001)

history = model.fit(data_generator.flow(x_train, y_train, batch_size=32), epochs=12, validation_data=data_generator.flow(x_val, y_val), callbacks=[learning_rate_reduction])


# %% saving the model

model.save("final_model.h5")

# %% print the accuracy and the loss

print("Loss of the model:", round(model.evaluate(x_test, y_test)[0], 2))
print("Accuracy of the model:", round(model.evaluate(x_test, y_test)[1] * 100, 2), "%")

# %% plot the accuracy and the loss

epochs = [i for i in range(12)]
fig, ax = plt.subplots(1,2)
train_accuracy = history.history["accuracy"]
train_loss = history.history["loss"]
validation_accuracy = history.history["val_accuracy"]
validation_loss = history.history["val_loss"]
fig.set_size_inches(20,10)

ax[0].plot(epochs, train_accuracy, "go-", label="Training Accuracy")
ax[0].plot(epochs, validation_accuracy, "ro-", label="Validation Accuracy")
ax[0].set_title("Training & Validation Accuracy")
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, "g-o", label="Training Loss")
ax[1].plot(epochs, validation_loss, "r-o", label="Validation Loss")
ax[1].set_title("Testing Accuracy & Loss")
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()

# %% load the model and make predictions

model_1 = load_model("final_model.h5")

probabilities = model_1.predict(x_test)
predictions = (probabilities > 0.6).astype(int)
# predictions = np.argmax(probabilities, axis=1)
predictions = predictions.reshape(1,-1)[0]

# %%

print(classification_report(y_test, predictions, target_names = ["Pneumonia (Class 0)", "Normal (Class 1)"], zero_division=1))

# %%

cm = confusion_matrix(y_test, predictions)
print(cm)

# %%

cm = pd.DataFrame(cm , index = ["0","1"] , columns = ["0","1"])

plt.figure(figsize = (10,10))
sns.heatmap(cm, cmap="Blues", linecolor="black", linewidth=1, annot=True, fmt="", xticklabels=labels, yticklabels=labels)
plt.show()

# %%

correct = np.nonzero(predictions == y_test)[0]
incorrect = np.nonzero(predictions != y_test)[0]

# %%

i = 0
for c in correct[:6]:
    plt.subplot(4,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[c].reshape(150,150), cmap="gray", interpolation="none")
    plt.title("Predicted Class {}, Actual Class {}".format(predictions[c], y_test[c]))
    plt.tight_layout()
    i += 1
    
plt.show()
    
# %%

i = 0
for c in incorrect[:6]:
    plt.subplot(4,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[c].reshape(150,150), cmap="gray", interpolation="none")
    plt.title("Predicted Class {}, Actual Class {}".format(predictions[c], y_test[c]))
    plt.tight_layout()
    i += 1
    
plt.show()
