import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers import Conv2D, UpSampling2D,MaxPooling2D
import os

# Load and preprocess your data
data_dir = r'C:\Users\User\OneDrive\Desktop\data'
image_files = os.listdir(data_dir)
images = []
for image_file in image_files:
    image = cv2.imread(os.path.join(data_dir, image_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image = cv2.resize(image, (256, 256))
    images.append(image)
    print('loading...')
images = np.array(images)
images = images.astype('float32') / 255.0

# Split your data into a training set and a validation set
x_train, x_val, y_train, y_val = train_test_split(images[:,:,:,0], images[:,:,:,1:], test_size=0.2, random_state=42)

# Add an extra dimension to the inputs
x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)

inputs = Input(shape=(256, 256, 1))

# Encoder
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

# Decoder
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)

conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)

# Output layer
out = Conv2D(2, (3, 3), activation='tanh', padding='same')(up2)

model = Model(inputs=inputs, outputs=out)

# Compile your model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train your model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=150, batch_size=64)

model.save('colorize_opencv.keras')
