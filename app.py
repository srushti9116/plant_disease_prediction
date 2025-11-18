import os
import json
import copy
import numpy as np
import datetime
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import plotly.graph_objects as go
import plotly.express as px
from pdf_generator import generate_pdf_report
from delivery import mail, deliver_report

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['REPORTS_FOLDER'] = os.path.join('static', 'reports')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['REPORTS_FOLDER']):
    os.makedirs(app.config['REPORTS_FOLDER'])

# Configure Flask-Mail
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

mail.init_app(app)

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
    """Load image, preprocess, run prediction + return top 5 labels and confidences."""
    target_size = (224, 224)  # adjust if your model uses other size
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]

    # Get top 5 indices and confidences
    top5_indices = np.argsort(preds)[-5:][::-1]  # Sort descending
    top5_confidences = [float(conf) for conf in preds[top5_indices] * 100.0]
    top5_labels = [idx_to_class.get(int(idx), "Unknown") for idx in top5_indices]

    # For backward compatibility, return top1 as primary
    top1_index = top5_indices[0]
    top1_confidence = top5_confidences[0]
    top1_label = top5_labels[0]

    print("=== PREDICTION ===")
    print("Top 1 - Index:", top1_index, "Label:", repr(top1_label), "Confidence:", top1_confidence)
    for i, (label, conf) in enumerate(zip(top5_labels, top5_confidences), 1):
        print(f"Top {i} - Label: {repr(label)}, Confidence: {conf:.2f}")

    # Return top1 and top5 list
    top5_candidates = list(zip(top5_labels, top5_confidences))
    return top1_label, top1_confidence, top5_candidates
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

def store_prediction(prediction_data, image_path):
    """Store prediction data in JSON file."""
    try:
        with open('predictions.json', 'r') as f:
            predictions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        predictions = []

    prediction_entry = {
        'id': len(predictions) + 1,
        'timestamp': datetime.datetime.now().isoformat(),
        'plant': prediction_data['plant'],
        'disease': prediction_data['disease'],
        'confidence': prediction_data['confidence'],
        'severity': prediction_data['severity'],
        'image_path': image_path.replace("\\", "/")
    }

    predictions.append(prediction_entry)

    with open('predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)

def get_dashboard_data():
    """Load and process prediction data for dashboard."""
    try:
        with open('predictions.json', 'r') as f:
            predictions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        predictions = []

    # Process data for visualizations
    disease_counts = {}
    confidence_data = []
    plant_counts = {}
    severity_counts = {}

    for pred in predictions:
        disease = pred['disease']
        confidence = pred['confidence']
        plant = pred['plant']
        severity = pred['severity']

        disease_counts[disease] = disease_counts.get(disease, 0) + 1
        plant_counts[plant] = plant_counts.get(plant, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        confidence_data.append(confidence)

    return {
        'total_predictions': len(predictions),
        'disease_counts': disease_counts,
        'plant_counts': plant_counts,
        'severity_counts': severity_counts,
        'confidence_data': confidence_data
    }



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
        label, confidence, top5_candidates = model_predict(save_path)
        plant, disease_readable = parse_label(label)

        # Get disease info & remedies
        details = get_details_for_label(label, plant)
        remedies = details.get("remedies", [])

        # Debug
        print("Using disease pattern for label:", label)
        print("Plant:", plant, "| Disease:", disease_readable)
        print("Number of remedies:", len(remedies))

        image_web_path = f"/static/uploads/{filename}"

        # Prepare prediction data for storage and PDF
        prediction_data = {
            'plant': plant,
            'disease': disease_readable,
            'confidence': round(confidence, 2),
            'description': details["description"],
            'symptoms': details["symptoms"],
            'management': details["management"],
            'severity': details["severity"],
            'remedies': remedies,
            'top5_candidates': top5_candidates
        }

        # Store prediction in JSON
        store_prediction(prediction_data, save_path)

        # Generate PDF report
        pdf_filename = f"report_{int(datetime.datetime.now().timestamp())}.pdf"
        pdf_path = os.path.join(app.config['REPORTS_FOLDER'], pdf_filename)
        generate_pdf_report(prediction_data, save_path, pdf_path)

        # Trigger multi-channel delivery
        contact_info = {
            'whatsapp': request.form.get('whatsapp'),
            'sms': request.form.get('sms'),
            'email': request.form.get('email')
        }
        delivery_status = []
        if any(contact_info.values()):
            delivery_status = deliver_report(prediction_data, pdf_path, contact_info)

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
            remedies=remedies,
            top5_candidates=top5_candidates,
            pdf_path=f"/static/reports/{pdf_filename}",
            delivery_status=delivery_status
        )

    return "Something went wrong. Please try again."


@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/dashboard')
def dashboard():
    dashboard_data = get_dashboard_data()

    # Create Plotly figures
    # Disease frequency chart
    disease_fig = go.Figure(data=[
        go.Bar(
            x=list(dashboard_data['disease_counts'].keys()),
            y=list(dashboard_data['disease_counts'].values()),
            marker_color='lightblue'
        )
    ])
    disease_fig.update_layout(
        title="Disease Frequency",
        xaxis_title="Disease",
        yaxis_title="Count",
        template="plotly_white"
    )

    # Plant distribution chart
    plant_fig = go.Figure(data=[
        go.Pie(
            labels=list(dashboard_data['plant_counts'].keys()),
            values=list(dashboard_data['plant_counts'].values()),
            hole=0.3
        )
    ])
    plant_fig.update_layout(
        title="Plant Distribution",
        template="plotly_white"
    )

    # Confidence distribution histogram
    confidence_fig = go.Figure(data=[
        go.Histogram(
            x=dashboard_data['confidence_data'],
            nbinsx=20,
            marker_color='green'
        )
    ])
    confidence_fig.update_layout(
        title="Prediction Confidence Distribution",
        xaxis_title="Confidence (%)",
        yaxis_title="Frequency",
        template="plotly_white"
    )

    # Severity distribution
    severity_fig = go.Figure(data=[
        go.Bar(
            x=list(dashboard_data['severity_counts'].keys()),
            y=list(dashboard_data['severity_counts'].values()),
            marker_color='orange'
        )
    ])
    severity_fig.update_layout(
        title="Disease Severity Distribution",
        xaxis_title="Severity",
        yaxis_title="Count",
        template="plotly_white"
    )

    # Convert to HTML
    disease_chart = disease_fig.to_html(full_html=False)
    plant_chart = plant_fig.to_html(full_html=False)
    confidence_chart = confidence_fig.to_html(full_html=False)
    severity_chart = severity_fig.to_html(full_html=False)

    return render_template(
        'dashboard.html',
        total_predictions=dashboard_data['total_predictions'],
        disease_chart=disease_chart,
        plant_chart=plant_chart,
        confidence_chart=confidence_chart,
        severity_chart=severity_chart
    )


if __name__ == '__main__':
    app.run(debug=True)
