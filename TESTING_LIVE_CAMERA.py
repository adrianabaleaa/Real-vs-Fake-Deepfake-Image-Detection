import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


# 1. SETUP & MODEL LOADING (Bulletproof)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'deepfake_detector.keras')

print(f"Loading model from: {model_path}")
print("Please wait...")

# (EfficientNetB0 architecture)
base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Load the weights into the skeleton
try:
    model.load_weights(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {e}")
    exit()



# 2. HELPER FUNCTION FOR ANALYSIS
def analyze_snapshot(frame_to_test):
    """
    Takes a single frame (BGROpenCV image), 
    processes it, and returns the AI prediction.

    """
    # 1. Convert OpenCV BGR to RGB
    rgb_frame = cv2.cvtColor(frame_to_test, cv2.COLOR_BGR2RGB)
    
    # 2. Resize to EfficientNet input size (224x224)
    resized_frame = cv2.resize(rgb_frame, (224, 224))
    
    # 3. Convert to numpy array and add batch dimension
    img_array = image.img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Run the AI prediction (silent mode)
    prediction = model.predict(img_array, verbose=0)
    score = prediction[0][0]
    return score



# 3. MAIN APPLICATION LOOP
print("1. Look at the camera for the live preview.")
print("2. Press 'S' to take a snapshot & analyze it.")
print("3. On the result screen, press 'R' to return to live video.")
print("4. Press 'Q' to quit anytime.")


# Open webcam (0 is default)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Set window name
window_name = 'Deepfake Detector - Snap & Test'
cv2.namedWindow(window_name)

# Application States
result_image = None
showing_result = False

while True:
    if not showing_result:
        
        # LIVE PREVIEW 
        ret, frame = cap.read()
        if not ret: break

        # Add instruction overlay to live feed
        preview_frame = frame.copy()
        cv2.putText(preview_frame, "LIVE PREVIEW - Press 'S' to Snap", (10, preview_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow(window_name, preview_frame)
        
        # Listen for keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("Snapshot taken! Analyzing image...")
            # Capture the current live frame for testing
            captured_frame = frame.copy()
            
            # RUN AI ANALYSIS on this single frame
            final_score = analyze_snapshot(captured_frame)
            
            # Prepare the result static image
            result_image = captured_frame.copy()
            
            if final_score > 0.5:
                text = f"REAL: {(final_score * 100):.1f}%"
                color = (0, 255, 0) # Green
            else:
                text = f"FAKE/AI: {((1 - final_score) * 100):.1f}%"
                color = (0, 0, 255) # Red
            
            # Draw result overlay on the static image
            h, w, _ = result_image.shape
            cv2.rectangle(result_image, (0, 0), (w, 60), (0,0,0), -1) # Black header bar
            cv2.putText(result_image, text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
            cv2.putText(result_image, "Result - Press 'R' to Return or 'Q' to Quit", (10, result_image.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            print(f"Analysis complete: {text}")
            showing_result = True # Switch state to show static image
            
        elif key == ord('q'):
            break

    else:
        
        # DISPLAY STATIC RESULT
        cv2.imshow(window_name, result_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            print("Returning to live preview...")
            showing_result = False # Switch back to live state
        elif key == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Application closed successfully.")