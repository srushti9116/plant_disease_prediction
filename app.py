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

model = load_model(MODEL_PATH, compile=False)


with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)

# Example: {"Apple___Apple_scab": 0, ...} -> invert to index -> label
idx_to_class = {v: k for k, v in class_indices.items()}

print("Model loaded. Total classes:", len(idx_to_class))


# ---------- DEFAULT DETAILS (for unknown label) ----------

DEFAULT_DETAILS = {
    "description": (
        "The system detected some irregular patterns on the leaf, but it does not clearly "
        "match any specific known disease category. This may be due to mild stress, poor "
        "image clarity, or early-stage symptoms."
    ),
    "symptoms": (
        "Minor discoloration, uneven texture, nutrient stress marks, or early signs of fungal "
        "or bacterial infection that are not yet distinct enough for precise identification."
    ),
    "management": (
        "Keep monitoring the plant for any increase in symptoms. Ensure proper sunlight, good "
        "soil drainage, balanced fertilization, and avoid wetting the foliage late in the day. "
        "If symptoms spread quickly, consult a local agriculture expert for accurate diagnosis."
    ),
    "severity": "Low",
    "remedies": [
        {
            "name": "General Organic Plant Tonic",
            "description": "Helps boost plant immunity and overall leaf health. Suitable for early or unclear symptoms.",
            "image": "/static/remedies/general_tonic.jpg",
            "link": "https://www.amazon.in/s?k=plant+growth+organic+tonic"
        },
        {
            "name": "Neem Oil (Organic Pesticide)",
            "description": "Useful as a preventive measure against common fungal and insect problems.",
            "image": "/static/remedies/neem_oil.jpg",
            "link": "https://www.flipkart.com/search?q=neem+oil+for+plants"
        }
    ]
}



# ---------- DISEASE-WISE INFO (matched by substring in label) ----------
DISEASE_KB = {
    # HEALTHY pattern (works for Tomato___healthy, Potato___Healthy, etc.)
    "healthy": {
        "description": "{plant} leaf appears healthy. No major disease symptoms are visible based on the image.",
        "symptoms": (
            "Uniform green colour, no obvious spots, no yellow halos, no mold or rotting area. "
            "Minor dust or tiny marks are normal in field conditions."
        ),
        "management": (
            "Continue regular care: proper watering, balanced fertilizer, and timely weeding. "
            "Keep checking plants once a week so that any future disease can be caught early."
        ),
        "severity": "None",
        "remedies": []
    },

    "Early_blight": {
        "description": "{plant} early blight is a common fungal disease usually caused by Alternaria species.",
        "symptoms": (
            "Dark brown concentric target-like spots on older leaves, yellowing around the lesions, "
            "starting from lower leaves and slowly moving upward."
        ),
        "management": (
            "Collect and destroy infected leaves, avoid overhead irrigation, rotate crops and do not plant "
            "tomato/potato in the same spot every year. Use disease-free seedling stock."
        ),
        "severity": "High",
        "remedies": [
            {
                "name": "Early blight fungicide spray",
                "description": "Protectant or systemic fungicide used to manage early blight in tomato and potato. Use strictly as per product label.",
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
        "symptoms": (
            "Water-soaked areas on leaves which turn dark brown to black, sometimes with white fluffy growth "
            "on the underside under humid conditions. Foliage may suddenly collapse."
        ),
        "management": (
            "Immediately remove and destroy badly infected plants, avoid wetting foliage in late evening and "
            "ensure good drainage. Plant resistant or tolerant varieties wherever available."
        ),
        "severity": "Very High",
        "remedies": [
            {
                "name": "Systemic fungicide for late blight",
                "description": "Broad-spectrum systemic fungicide often recommended for late blight management. Follow local guidelines and label instructions.",
                "image": "/static/remedies/tomato_late_blight_tonic.jpg",
                "link": "https://www.amazon.in/s?k=late+blight+fungicide"
            },
            {
                "name": "Protective contact fungicide",
                "description": "Contact fungicide sprayed as a preventive in disease-prone, cool and wet conditions.",
                "image": "/static/remedies/contact_fungicide.jpg",
                "link": "https://www.flipkart.com/search?q=contact+fungicide+for+plants"
            }
        ]
    },

    "Leaf_Mold": {
        "description": "{plant} leaf mold is a fungal disease that prefers high humidity and poor air circulation.",
        "symptoms": (
            "Yellow spots on the upper leaf surface with olive-green to brown velvety mold on the underside. "
            "Heavily infected leaves may curl and dry."
        ),
        "management": (
            "Improve ventilation in the field or polyhouse, avoid overcrowding of plants, remove affected leaves "
            "and avoid frequent light irrigations that keep leaves continuously wet."
        ),
        "severity": "Medium",
        "remedies": [
            {
                "name": "Leaf mold control spray",
                "description": "Fungicide spray generally recommended for leaf mold in tomato and other crops.",
                "image": "/static/remedies/leaf_mold_spray.jpg",
                "link": "https://www.amazon.in/s?k=leaf+mold+fungicide"
            }
        ]
    },

    "Bacterial_spot": {
        "description": "{plant} bacterial spot is caused by Xanthomonas species and affects leaves and sometimes fruits.",
        "symptoms": (
            "Small water-soaked spots that later turn dark and may crack. In severe cases leaves turn yellow and drop, "
            "leading to a scorched appearance."
        ),
        "management": (
            "Use certified disease-free seed, avoid overhead irrigation and working in the field when foliage is wet. "
            "Remove and destroy infected plant residues after harvest."
        ),
        "severity": "Medium",
        "remedies": [
            {
                "name": "Copper-based bactericide spray",
                "description": "Copper formulations usually recommended against bacterial leaf spot on many crops.",
                "image": "/static/remedies/pepper_bacterial_spot_spray.jpg",
                "link": "https://www.flipkart.com/search?q=copper+spray+for+plants"
            }
        ]
    },

    "Apple_scab": {
        "description": "Apple scab is a fungal disease of apple caused by Venturia inaequalis.",
        "symptoms": (
            "Olive-green to dark brown velvety spots on leaves and fruits. Leaves may twist and fall prematurely; "
            "fruits may crack or become deformed."
        ),
        "management": (
            "Prune trees to improve air movement, rake and destroy fallen leaves, and follow a recommended fungicide "
            "spray schedule during the critical infection period."
        ),
        "severity": "Medium",
        "remedies": [
            {
                "name": "Apple scab fungicide",
                "description": "Fungicide products labeled specifically for apple scab management.",
                "image": "/static/remedies/apple_scab_spray.jpg",
                "link": "https://www.amazon.in/s?k=apple+scab+fungicide"
            }
        ]
    },

    "Black_rot": {
        "description": "Black rot is a fungal disease that can affect leaves, fruits and stems of {plant}.",
        "symptoms": (
            "Dark brown to black lesions on leaves, often with concentric ring patterns. On fruits, dark rot develops "
            "and in grapes berries may shrivel into hard mummies."
        ),
        "management": (
            "Remove and destroy infected prunings and mummified fruits, improve air circulation by proper pruning, "
            "and avoid overhead irrigation late in the day."
        ),
        "severity": "High",
        "remedies": [
            {
                "name": "Black rot control fungicide",
                "description": "Fungicide products labeled for black rot disease management on fruits and vines.",
                "image": "/static/remedies/grape_black_rot_spray.jpg",
                "link": "https://www.amazon.in/s?k=black+rot+fungicide"
            }
        ]
    }
}

def get_details_for_label(label: str, plant: str):
    """
    1. Look for any DISEASE_KB key that appears as substring in the raw label.
       (e.g. 'Early_blight' inside 'Tomato___Early_blight')
    2. If matched, format text with {plant}.
    3. Otherwise return DEFAULT_DETAILS.
    """
    label_lower = label.lower()

    for pattern, template in DISEASE_KB.items():
        if pattern.lower() in label_lower:
            details = copy.deepcopy(template)

            # Format strings containing {plant}
            for key, value in details.items():
                if isinstance(value, str):
                    details[key] = value.format(plant=plant)

                if key == "remedies":
                    for item in details["remedies"]:
                        for k2, v2 in item.items():
                            if isinstance(v2, str):
                                item[k2] = v2.format(plant=plant)

            return details

    # No pattern matched → generic info
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
# ---------- SMALL HELPER ----------
def parse_label(label: str):
    """
    Split 'Tomato___Early_blight' into:
        plant='Tomato', disease='Early blight'
    and make them human-readable.
    """
    if "___" in label:
        plant, disease = label.split("___", 1)
    else:
        plant, disease = "Unknown", label

    plant_readable = plant.replace("_", " ")
    disease_readable = disease.replace("_", " ")
    return plant_readable, disease_readable



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
