from keras.models import load_model
import numpy as np
import cv2

# Load the pre-trained model
model = load_model('colorize_opencv.keras')

# Use your trained model to colorize images
color_image = cv2.imread(r'C:\Users\User\OneDrive\Desktop\download.jpeg')
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2Lab)
color_image = cv2.resize(color_image, (256, 256))
l_channel = color_image[:,:,0]
l_channel = l_channel.astype('float32') / 255.0
l_channel = np.expand_dims(l_channel, axis=-1)
l_channel = np.expand_dims(l_channel, axis=0)  # add an extra dimension for the batch size
ab_channels = model.predict(l_channel)

# Ensure that l_channel and ab_channels both have 3 dimensions
l_channel = np.squeeze(l_channel, axis=0)
ab_channels = np.squeeze(ab_channels, axis=0)

colorized_image = np.concatenate((l_channel, ab_channels), axis=-1)
colorized_image = colorized_image * 255.0  # denormalize the pixel values
colorized_image = colorized_image.astype('uint8')  # convert to integer pixel values

# Ensure that colorized_image has 3 channels before converting color spaces
if colorized_image.shape[-1] == 2:
    colorized_image = np.concatenate((colorized_image, np.zeros((colorized_image.shape[0], colorized_image.shape[1], 1))), axis=-1)

colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_Lab2BGR)

# Display the grayscale image
grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_Lab2BGR)
cv2.imshow('Grayscale Image', grayscale_image)


# Display the colorized image
cv2.imshow('Colorized Image', colorized_image)
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()