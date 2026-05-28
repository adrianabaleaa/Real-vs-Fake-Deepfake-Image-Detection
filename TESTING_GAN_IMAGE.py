import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


# 1. SETUP PATHS
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'deepfake_detector.keras')

# Image 
image_filename = 'test.jpg' 
image_path = os.path.join(current_dir, image_filename)

print(f"Preparing to test image: {image_filename}")


# 2. LOAD MODEL (The bulletproof way)
print("2. Rebuilding network and loading AI brain...")
base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

try:
    model.load_weights(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()


# 3. LOAD AND ANALYZE THE IMAGE
print("Analyzing the image...")

# Read the image using OpenCV
img = cv2.imread(image_path)

if img is None:
    print(f"ERROR: Could not find '{image_filename}' in the folder!")
    print(f"Make sure you placed the picture in: {current_dir}")
    exit()

# Process image for the AI (BGR to RGB, then resize to 224x224)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_img = cv2.resize(rgb_img, (224, 224))

# Convert to array and add batch dimension
img_array = image.img_to_array(resized_img)
img_array = np.expand_dims(img_array, axis=0)

# Make the prediction
prediction = model.predict(img_array, verbose=0)
score = prediction[0][0]


# 4. DISPLAY THE RESULT 
# Determine label and color
if score > 0.5:
    text = f"REAL: {(score * 100):.1f}%"
    color = (0, 255, 0) # Green
    print(f"\n---> ACCURACY: {text} <---")
else:
    text = f"FAKE/AI: {((1 - score) * 100):.1f}%"
    color = (0, 0, 255) # Red
    print(f"\n---> ACCURACY: {text} <---")

# Draw the result on the original image.
cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0,0,0), -1) 
cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
cv2.namedWindow('Static Image Test', cv2.WINDOW_AUTOSIZE)

# Show the image
cv2.imshow('Static Image Test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()