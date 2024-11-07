import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array, load_img
import matplotlib.pyplot as plt


# Load the model for bone fracture 
model_path = '/content/drive/MyDrive/final_prjct/model (1).h5'
model_fracture = load_model(model_path)

# Mapping indices to classes for bone fracture prediction
class_names = ['elbow_negative', 'elbow_positive', 'finger_negative', 'finger_positive',
               'forearm_negative', 'forearm_positive', 'hand_negative', 'hand_positive',
               'humerus_negative', 'humerus_positive', 'shoulder_negative', 'shoulder_positive',
               'wrist_negative', 'wrist_positive']

# Load the YOLO model for object detection
model_yolo = YOLO('/content/drive/MyDrive/final_prjct/best.pt')

# Streamlit app

st.markdown("<h1 style='text-align: center;'>Bone Fracture Detection</h1>", unsafe_allow_html=True) # Center-align the title

# Types of bone classification heading
st.markdown("<h2 style='text-align: center;'>Types of Bone Classification</h2>", unsafe_allow_html=True)

image_path = "/content/drive/MyDrive/final_prjct/Screenshot (228).png"  # Provide the path to your image

# Load the image
uploaded_image = Image.open(image_path)

# Display the uploaded image
st.image(uploaded_image, use_column_width=True)

# Split the interface into two equal columns
left_column, right_column = st.columns(2)

# File uploader for image in the left column
left_column.title('Upload Image')
right_column.title('Output Image')
uploaded_file = left_column.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the uploaded image for object detection
    image = Image.open(uploaded_file)
    left_column.image(image, caption='Uploaded Image', use_column_width=True)
    img = image.resize((224, 224))  # Resize the image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Submit button
    if left_column.button('Submit'):
        y_pred = model_fracture.predict(img_array)
        prediction_index = np.argmax(y_pred, axis=1)[0]
        op1, op2 = class_names[prediction_index].split('_')
      
        # Perform object detection using YOLO
        results = model_yolo(image)

        # Check if any bounding boxes are detected
        bbox_detected = any(len(r.boxes) > 0 for r in results)
        right_column.markdown("<h4>The uploaded image seems to be: {}</h4>".format(op1), unsafe_allow_html=True)
         

        if not bbox_detected:
            right_column.markdown("<div style='border: 1px solid #ccc; padding: 10px; text-align: center; font-size: larger;'>No fractures detected</div>", unsafe_allow_html=True)
        else:
            right_column.markdown("<h3>Fracture(s) detected</h3>", unsafe_allow_html=True)
           
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                right_column.image(im, caption='Detected Fracture', use_column_width=True)
