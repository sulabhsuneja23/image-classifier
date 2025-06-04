from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
# Load model
model = load_model("classifier_model.h5")

# Class labels (manually defined or loaded)
class_labels = ["class1", "class2", "class3", "class4"]

folder_path = "./testdata"

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        file_path = os.path.join(folder_path, filename)
        # Load and preprocess image
        
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)
        predicted_class = np.argmax(pred)
        print( "File:", filename)   
        print("Predicted class:", class_labels[predicted_class])