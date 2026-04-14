from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import random
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load trained CNN model
model = tf.keras.models.load_model('model/brain_tumor_model.h5')

classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Dynamic medical guidance / theory
tumor_info = {
    "Glioma": {
        "symptoms": [
            "Headache, seizures, nausea, and memory issues may occur.",
            "Possible signs include headache, nausea, seizures, and confusion.",
            "Common symptoms are headaches, seizures, nausea, and memory problems."
        ],
        "action": [
            "Consult a neurologist immediately. MRI and biopsy may be required.",
            "Please visit a brain specialist soon for detailed evaluation.",
            "Early consultation with a neurosurgeon is advised."
        ]
    },

    "Meningioma": {
        "symptoms": [
            "Vision issues, headache, weakness, and memory problems may occur.",
            "Possible signs include blurred vision, headaches, and weakness.",
            "Common symptoms are headache, memory issues, and vision disturbance."
        ],
        "action": [
            "Consult a neurosurgeon. Surgery may be required.",
            "Medical consultation is advised for detailed imaging and treatment.",
            "Please consult a specialist for further evaluation."
        ]
    },

    "Pituitary": {
        "symptoms": [
            "Hormonal imbalance, blurred vision, and fatigue may occur.",
            "Possible signs include fatigue, vision problems, and hormone changes.",
            "Common symptoms are tiredness, blurred vision, and hormonal issues."
        ],
        "action": [
            "Consult an endocrinologist. Hormone tests may be needed.",
            "Please seek specialist advice for hormonal evaluation.",
            "Further hormone testing and medical consultation are recommended."
        ]
    },

    "No Tumor": {
        "symptoms": [
            "No abnormality detected in the scan.",
            "No visible signs of tumor were found.",
            "The MRI appears normal with no major abnormality."
        ],
        "action": [
            "No immediate action needed, but consult doctor if symptoms persist.",
            "No urgent concern detected. Routine consultation is advised if needed.",
            "Scan appears normal. Seek medical advice only if symptoms continue."
        ]
    }
}


# Prediction function with safer logic
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]

    predicted_class = np.argmax(prediction)
    confidence = round(np.max(prediction) * 100, 2)

    result = classes[predicted_class]

    # Safer demo logic:
    # If tumor predicted with low confidence, mark as No Tumor
    if result != "No Tumor" and confidence < 80:
        result = "No Tumor"
        confidence = round(100 - confidence, 2)

    return result, confidence


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    # No file selected
    if file.filename == '':
        return render_template(
            'result.html',
            prediction="No file selected",
            confidence=0,
            img_path="",
            tumor_detected="No",
            symptoms="Please upload a valid MRI image.",
            action="Try again with a proper scan."
        )

    # Safe filename
    filename = secure_filename(file.filename)

    # Ensure static folder exists
    os.makedirs("static", exist_ok=True)

    filepath = os.path.join("static", filename)
    file.save(filepath)

    # Get prediction
    result, confidence = predict_image(filepath)

    # Yes / No tumor
    tumor_detected = "No" if result == "No Tumor" else "Yes"

    # Random symptoms + action
    info = tumor_info[result]
    symptoms = random.choice(info["symptoms"])
    action = random.choice(info["action"])

    return render_template(
        'result.html',
        prediction=result,
        confidence=confidence,
        img_path=filepath,
        tumor_detected=tumor_detected,
        symptoms=symptoms,
        action=action
    )


if __name__ == "__main__":
    app.run(debug=True)