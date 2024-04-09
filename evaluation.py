import json
import numpy as np
import os
from keras.preprocessing import image

from src.models.initialModel import trained_model  # TODO change to the actual trained model function

# Load the model
model = trained_model()  # TODO change to the actual trained model function

# Load the validation images from the directory
validation_dir = '../validation/rgb/' 
validation_images = [os.path.join(validation_dir, img) for img in os.listdir(validation_dir) if img.endswith('.jpg')]

# Process the validation images
results = []
for img_path in validation_images:
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(340, 240))  # TODO Replace with our target size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)

    # Generate output in the json format
    result = {
        'image_id': img_path,
        'parameters': prediction.tolist()   #TODO check wether parameters is sufficient to set the output or if we need the names as in the json
    }
    results.append(result)

# Write the results to the submission file
submission_file = os.path.join(validation_dir, 'validation_submission.json')
os.makedirs(os.path.dirname(submission_file), exist_ok=True)
with open(submission_file, 'w') as f:
    json.dump(results, f)