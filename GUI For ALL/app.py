from flask import Flask, render_template, request
import os
import math
import numpy as np
import cv2
import pytesseract
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import re
from functools import lru_cache

app = Flask(__name__)

# Load models
yolo_model = YOLO('date_9_model.pt')
freshness_model = tf.keras.models.load_model('fruit_freshness_model.h5')

# Define class labels for freshness prediction
class_labels = [
    'Bad Apple', 'Bad Banana', 'Bad Bellpepper', 'Bad Cucumber', 'Bad Grapes',
    'Bad Indian Green Chile', 'Bad Mango', 'Bad Orange', 'Bad Potato', 'Bad Tomato',
    'Fresh Apple', 'Fresh Banana', 'Fresh Bellpepper', 'Fresh Cucumber', 'Fresh Grapes',
    'Fresh Indian Green Chile', 'Fresh Mango', 'Fresh Orange', 'Fresh Potato', 'Fresh Tomato',
    'Moderate Apple', 'Moderate Banana', 'Moderate Bellpepper', 'Moderate Cucumber',
    'Moderate Grapes', 'Moderate Indian Green Chile', 'Moderate Mango', 'Moderate Orange',
    'Moderate Potato', 'Moderate Tomato'
]

# Arrhenius equation-related factors
PreExponentialFactor = {
    'Fresh Apple': 645092.13, 'Fresh Banana': 15370.17, 'Fresh Bellpepper': 23330.75,
    'Fresh Cucumber': 277.94, 'Fresh Grapes': 23330.75, 'Fresh Indian Green Chile': 23330.75,
    'Fresh Mango': 277.94, 'Fresh Orange': 5.99, 'Fresh Potato': 295.32,
    'Fresh Tomato': 15370.17, 'Moderate Apple': 15370.17, 'Moderate Banana': 11139.42,
    'Moderate Bell Pepper': 113.08, 'Moderate Cucumber': 3494.56, 'Moderate Grapes': 440004.84,
    'Moderate Indian Green Chile': 113.08, 'Moderate Mango': 208.10, 'Moderate Orange': 277.94,
    'Moderate Potato': 1747.28, 'Moderate Tomato': 440004.84
}
ActivationEnergy = {
    'Fresh Apple': 38867.22, 'Fresh Banana': 27885.78, 'Fresh Bellpepper': 29754.35,
    'Fresh Cucumber': 18772.91, 'Fresh Grapes': 29754.35, 'Fresh Indian Green Chile': 29754.35,
    'Fresh Mango': 18772.91, 'Fresh Orange': 10981.45, 'Fresh Potato': 20641.48,
    'Fresh Tomato': 27885.78, 'Moderate Apple': 27885.78, 'Moderate Banana': 24816.43,
    'Moderate Bell Pepper': 15156.40, 'Moderate Cucumber': 22947.86, 'Moderate Grapes': 33929.30,
    'Moderate Indian Green Chile': 15156.40, 'Moderate Mango': 16191.56, 'Moderate Orange': 18772.91,
    'Moderate Potato': 22947.86, 'Moderate Tomato': 33929.30
}

# Shelf-life calculation
def calculate_rate_constant(Ea, A, T):
    R = 8.314  # Universal gas constant in J/(mol*K)
    k = A * math.exp(-Ea / (R * T))
    return k

def predict_shelf_life(Ea, A, T):
    k = calculate_rate_constant(Ea, A, T)
    shelf_life = 1 / k
    return shelf_life

def predict_image(image_path, temperature_celsius):
    if not os.path.exists(image_path):
        return None

    try:
        # Load and preprocess image
        img = Image.open(image_path)
        img_resized = img.resize((224, 224))  # Resize to match the model's input shape
        img_array = np.array(img_resized) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = freshness_model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]

        if predicted_class.startswith("Bad"):
            return predicted_class, 0  # Bad items have a shelf life of 0
        else:
            # Convert Celsius to Kelvin
            T_kelvin = temperature_celsius + 273.15

            # Calculate shelf life for fresh or moderate fruits
            Ea = ActivationEnergy[predicted_class]
            A = PreExponentialFactor[predicted_class]
            shelf_life = predict_shelf_life(Ea, A, T_kelvin)

            return predicted_class, shelf_life

    except Exception as e:
        print(f"Error in predicting image: {e}")
        return None

# IR counting
def perform_ir_counting(image_path):
    # Predict with YOLO model
    results = yolo_model.predict(source=image_path)
    outputs = {}  # Dictionary to store counts of each detected class

    # Read the image using OpenCV (if you need to display or process the image further)
    img = cv2.imread(image_path)

    # Extract results and count occurrences for each class
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract box coordinates and class information
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get bounding box coordinates
            class_id = int(box.cls[0])  # Get class ID
            conf = box.conf[0]  # Confidence score
            class_name = yolo_model.names[class_id]  # Get class name from YOLO model

            # Update the count for the detected class
            if class_name in outputs:
                outputs[class_name] += 1  # Increment the count if class already exists
            else:
                outputs[class_name] = 1  # Initialize the count if class doesn't exist

         
            # Uncomment this part if you need it
            # color = (0, 255, 0)  # Green color for bounding box
            # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # cv2.putText(img, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Convert the dictionary to a list of formatted strings for display
    output_list = [f"Item: {class_name} - Count: {count}" for class_name, count in outputs.items()]

    return output_list

# OCR with improved processing
def process_image(image_path):
    original_image = cv2.imread(image_path)

    if original_image is None:
        raise ValueError("Could not read the image. Please check the file path.")

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)
    blurred_image = cv2.GaussianBlur(denoised_image, (5, 5), 0)
    final_thresholded_image = cv2.adaptiveThreshold(blurred_image, 255,
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)
    return final_thresholded_image

def extract_text(final_thresholded_image):
    text = pytesseract.image_to_string(final_thresholded_image)
    return text.strip()

def clean_text(text):
    replacements = {
        'O': '0', 'I': '1', 'Z': '2', 'l': '1', 'NETQUANT1TY': 'NETQUANTITY',
        'l': '1', 'g': 'g', 'Gr': 'g',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove unwanted characters but keep spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with a single space
    cleaned_text = cleaned_text.strip()

    return cleaned_text

def extract_insights(cleaned_text):
    insights = {
        "Manufacturing Date": "Not Found",
        "Expiry Date": "Not Found",
        "Price": "Not Found",
        "Weight": "Not Found",
    }

    weight_pattern = r'NETQUANTITY\s*[:\-]?\s*(\d+)\s*[gG]'
    price_pattern = r'MRP\s*[:\-]?\s*(\d+)'
    mfg_date_pattern = r'Mfg Date\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})'
    exp_date_pattern = r'Exp Date\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})'

    weight_match = re.search(weight_pattern, cleaned_text, re.IGNORECASE)
    if weight_match:
        insights["Weight"] = weight_match.group(1) + " g"

    price_match = re.search(price_pattern, cleaned_text, re.IGNORECASE)
    if price_match:
        insights["Price"] = "INR " + price_match.group(1)

    mfg_date_match = re.search(mfg_date_pattern, cleaned_text, re.IGNORECASE)
    if mfg_date_match:
        insights["Manufacturing Date"] = mfg_date_match.group(1)

    exp_date_match = re.search(exp_date_pattern, cleaned_text, re.IGNORECASE)
    if exp_date_match:
        insights["Expiry Date"] = exp_date_match.group(1)

    return insights

@lru_cache(maxsize=128)
def perform_ocr(image_path):
    try:
        # Step 1: Process the image (grayscale, denoise, threshold)
        final_thresholded_image = process_image(image_path)

        # Step 2: Extract text from the processed image
        extracted_text = extract_text(final_thresholded_image)

        # Step 3: Clean the extracted text
        cleaned_extracted_text = clean_text(extracted_text)

        # Step 4: Extract insights from cleaned text
        insights = extract_insights(cleaned_extracted_text)

        return insights
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_shelf_life', methods=['POST'])
def predict_shelf_life_route():
    image = request.files['image']
    temperature = float(request.form['temperature'])
    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)

    result = predict_image(image_path, temperature)

    return render_template('index.html', shelf_life_result=result)

@app.route('/ir_counting', methods=['POST'])
def ir_counting_route():
    image = request.files['image']
    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)

    result = perform_ir_counting(image_path)

    return render_template('index.html', ir_counting_result=result)

@app.route('/perform_ocr', methods=['POST'])
def perform_ocr_route():
    image = request.files['image']
    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)

    ocr_result = perform_ocr(image_path)

    return render_template('index.html', ocr_result=ocr_result)

if __name__ == '__main__':
    app.run(debug=True)
