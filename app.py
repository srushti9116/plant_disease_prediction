import os
import json
import copy
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ---------- MODEL & LABELS ----------
MODEL_PATH = os.path.join('model', 'plant_disease_model.h5')
CLASS_INDICES_PATH = os.path.join('model', 'class_indices.json')

model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)

# Example: {"Apple___Apple_scab": 0, ...} -> invert to index -> label
idx_to_class = {v: k for k, v in class_indices.items()}

print("Model loaded. Total classes:", len(idx_to_class))


# ---------- DEFAULT DETAILS ----------
DEFAULT_DETAILS = {
    "description": "AI model detected a plant condition. Please verify visually and consult an agriculture expert before applying any chemical treatment.",
    "symptoms": "Discoloration, spots, or unusual patterns may appear on the leaf surface.",
    "management": "Monitor the plant closely, remove severely affected leaves, and maintain proper watering, spacing, and nutrients.",
    "severity": "Medium",
    "remedies": []
}

# ---------- KNOWLEDGE BASE BY DISEASE PATTERN (SUBSTRING) ----------
# We match using parts like 'Early_blight', 'Late_blight', etc.
DISEASE_KB = {
    "Early_blight": {
        "description": "{plant} early blight is a common fungal disease usually caused by Alternaria species.",
        "symptoms": "Dark brown concentric spots on older leaves, yellowing around the lesions, starting from lower leaves.",
        "management": "Remove infected leaves, avoid overhead watering, rotate crops, and use recommended fungicides when infection is severe.",
        "severity": "High",
        "remedies": [
            {
                "name": "Early blight fungicide tonic",
                "description": "Common fungicide spray used to manage early blight in tomato and potato. Use strictly as per product label.",
                "image": "/static/remedies/tomato_early_blight_tonic.jpg",
                "link": "https://www.amazon.in/s?k=early+blight+fungicide"
            },
            {
                "name": "Neem oil plant spray",
                "description": "Organic neem oil that helps reduce fungal and insect problems on leaves.",
                "image": "/static/remedies/neem_oil.jpg",
                "link": "https://www.flipkart.com/search?q=neem+oil+for+plants"
            }
        ]
    },

    "Late_blight": {
        "description": "{plant} late blight is a destructive disease usually caused by Phytophthora infestans.",
        "symptoms": "Water-soaked lesions on leaves, white mold growth under humid conditions, rapid blackening and rotting of foliage.",
        "management": "Immediately remove and destroy infected plants, avoid wet foliage, and apply systemic + protective fungicides as per recommendations.",
        "severity": "Very High",
        "remedies": [
            {
                "name": "Systemic fungicide for late blight",
                "description": "Broad-spectrum systemic fungicide used to control late blight on {plant}.",
                "image": "/static/remedies/tomato_late_blight_tonic.jpg",
                "link": "https://www.amazon.in/s?k=late+blight+fungicide"
            },
            {
                "name": "Protective contact fungicide",
                "description": "Sprayed as a preventive measure in disease-prone areas.",
                "image": "/static/remedies/contact_fungicide.jpg",
                "link": "https://www.flipkart.com/search?q=contact+fungicide+for+plants"
            }
        ]
    },

    "Leaf_Mold": {
        "description": "{plant} leaf mold is a fungal disease that prefers high humidity and poor air circulation.",
        "symptoms": "Yellow spots on upper leaf surface with olive-green to brown velvety mold on the underside.",
        "management": "Improve ventilation, avoid overcrowding, remove affected leaves and use recommended fungicides if required.",
        "severity": "Medium",
        "remedies": [
            {
                "name": "Leaf mold control spray",
                "description": "Fungicide spray for leaf mold management on greenhouse and field crops.",
                "image": "/static/remedies/leaf_mold_spray.jpg",
                "link": "https://www.amazon.in/s?k=leaf+mold+fungicide"
            }
        ]
    },

    "Bacterial_spot": {
        "description": "{plant} bacterial spot is caused by Xanthomonas species and affects leaves and fruits.",
        "symptoms": "Small water-soaked spots that turn dark and sometimes crack; in severe cases leaves may drop.",
        "management": "Use disease-free seed, avoid overhead irrigation, and apply copper-based bactericides where recommended.",
        "severity": "Medium",
        "remedies": [
            {
                "name": "Copper-based bactericide",
                "description": "Copper sprays are usually recommended against bacterial spot on {plant}.",
                "image": "/static/remedies/pepper_bacterial_spot_spray.jpg",
                "link": "https://www.flipkart.com/search?q=copper+spray+for+plants"
            }
        ]
    },

    "Apple_scab": {
        "description": "Apple scab is a fungal disease caused by Venturia inaequalis.",
        "symptoms": "Olive-green to brown velvety spots on leaves and fruits, leaf distortion and premature leaf fall.",
        "management": "Prune to improve air circulation, remove fallen leaves, and apply fungicides as per schedule.",
        "severity": "Medium",
        "remedies": [
            {
                "name": "Apple scab control fungicide",
                "description": "Fungicide products marketed for apple scab management.",
                "image": "/static/remedies/apple_scab_spray.jpg",
                "link": "https://www.amazon.in/s?k=apple+scab+fungicide"
            }
        ]
    },

    "Black_rot": {
        "description": "Black rot is a fungal disease that can affect {plant} leaves, fruits and stems.",
        "symptoms": "Dark brown to black lesions; in grapes, berries shrivel into mummies. In apples, fruit rot with concentric rings may occur.",
        "management": "Remove mummified fruits and infected debris, prune infected twigs, and apply recommended fungicides.",
        "severity": "High",
        "remedies": [
            {
                "name": "Black rot fungicide",
                "description": "Fungicides labeled for black rot management on fruits and vines.",
                "image": "/static/remedies/grape_black_rot_spray.jpg",
                "link": "https://www.amazon.in/s?k=black+rot+fungicide"
            }
        ]
    }
}


# ---------- SMALL HELPER ----------
def parse_label(label: str):
    """Split 'Tomato___Early_blight' into plant='Tomato', disease='Early blight'."""
    if "___" in label:
        plant, disease = label.split("___", 1)
    else:
        plant, disease = "Unknown", label

    plant_readable = plant.replace("_", " ")
    disease_readable = disease.replace("_", " ")
    return plant_readable, disease_readable


def get_details_for_label(label: str, plant: str):
    """
    1. Try to match disease pattern by substring (Early_blight, Late_blight, etc.)
    2. If nothing matches, return DEFAULT_DETAILS.
    """
    # Try pattern matching
    for pattern, template in DISEASE_KB.items():
        if pattern in label:
            # Copy so we don't modify original
            details = copy.deepcopy(template)

            # Format strings with {plant}
            for key, value in details.items():
                if isinstance(value, str):
                    details[key] = value.format(plant=plant)
                # remedies is a list of dicts; we can also format inside
                if key == "remedies":
                    for item in details["remedies"]:
                        for k2, v2 in item.items():
                            if isinstance(v2, str):
                                item[k2] = v2.format(plant=plant)
            return details

    # If no pattern matched -> generic info
    return DEFAULT_DETAILS


def model_predict(img_path):
    """Load image, preprocess, run prediction + return raw label and confidence."""
    target_size = (224, 224)  # adjust if your model uses other size
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]

    class_index = int(np.argmax(preds))
    confidence = float(preds[class_index]) * 100.0
    label = idx_to_class.get(class_index, "Unknown")

    print("=== PREDICTION ===")
    print("Index:", class_index)
    print("Raw label from model:", repr(label))
    print("Confidence:", confidence)

    return label, confidence


# ---------- ROUTES ----------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part in request"

    file = request.files['image']

    if file.filename == '':
        return "No file selected"

    if file:
        filename = file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Predict
        label, confidence = model_predict(save_path)
        plant, disease_readable = parse_label(label)

        # Get disease info & remedies
        details = get_details_for_label(label, plant)
        remedies = details.get("remedies", [])

        # Debug
        print("Using disease pattern for label:", label)
        print("Plant:", plant, "| Disease:", disease_readable)
        print("Number of remedies:", len(remedies))

        image_web_path = save_path.replace("\\", "/")

        return render_template(
            'result.html',
            image_file=image_web_path,
            raw_label=label,
            plant=plant,
            disease=disease_readable,
            confidence=round(confidence, 2),
            description=details["description"],
            symptoms=details["symptoms"],
            management=details["management"],
            severity=details["severity"],
            remedies=remedies
        )

    return "Something went wrong. Please try again."


@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
