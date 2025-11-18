

🌱 Plant Disease Detection System

### *Deep Learning (EfficientNet-B0 + CBAM) | Flask Web App | PDF Reports | WhatsApp/SMS/Email Delivery | Dashboard Analytics*

---

## 📌 Overview

This project is a **complete AI-based plant disease detection system** that identifies plant diseases from leaf images using a **deep learning model (EfficientNet-B0 + CBAM attention module)**.

It provides:

✔ Highly accurate disease prediction
✔ Disease description, symptoms, management & remedies
✔ Automatically generated PDF report
✔ Multi-channel delivery (Email / WhatsApp / SMS)
✔ Dashboard for analytics & tracking
✔ Top-5 prediction candidates
✔ Storage of prediction history

This implementation is ideal for **AIML mini-projects**, academic demos, and agricultural automation systems.

---

# 🚀 Features

### 🔍 1. Disease Prediction

* Upload a leaf image → predicts **plant & disease**
* Uses CNN model trained on 38 classes
* Output includes:

  * Plant Name
  * Disease Name
  * Confidence Score
  * Severity
  * Symptoms
  * Management
  * Remedies (Organic + Chemical)

### 🔁 2. Top-5 Prediction Candidates

Displays the model’s top-5 disease guesses with probability (%) to improve interpretability.

---

### 📄 3. Automated PDF Report Generation

Each prediction generates a downloadable PDF including:

* Uploaded image
* Disease info
* Remedies
* Severity level
* Confidence chart
* Model metadata

---

### 📤 4. Multi-Channel Delivery

You can send the generated report via:

* **📧 Email**
* **📱 WhatsApp**
* **📩 SMS**

Configured using:

* Flask-Mail
* Fast2SMS API (or similar)
* WhatsApp cloud message API

---

### 📊 5. Interactive Dashboard

A powerful analytics dashboard built with **Plotly** that shows:

* Disease frequency
* Plant distribution
* Confidence distribution
* Severity distribution
* Total predictions made

---

### 🧠 6. EfficientNet-B0 + CBAM AI Model

* Trained on PlantVillage dataset
* Includes **CBAM (Convolutional Block Attention Module)**
* 99%+ accuracy
* Works on 38 classes

Model files:

```
model/
 ├── plant_disease_model.h5
 └── class_indices.json
```

---

# 📁 Project Structure

```
plant_disease_prediction/
│
├── app.py                           # Flask main app
├── pdf_generator.py                 # PDF report generator
├── delivery.py                      # Email / WhatsApp / SMS module
├── predictions.json                 # All stored predictions
│
├── model/
│     ├── plant_disease_model.h5     # Trained model
│     └── class_indices.json         # Label mapping
│
├── static/
│     ├── uploads/                   # User-uploaded images
│     ├── reports/                   # Generated PDF reports
│     ├── remedies/                  # Remedy images
│     └── styles.css
│
└── templates/
      ├── index.html
      ├── result.html
      ├── dashboard.html
      ├── about.html
      └── how_it_works.html
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd plant_disease_prediction
```

## 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate    # For Windows
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Add Model Files

Place the following files inside `model/`:

* `plant_disease_model.h5`
* `class_indices.json`

---

# 🔐 Configure Environment Variables

Create a `.env` file or set system environment variables:

```
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=yourgmail@gmail.com
MAIL_PASSWORD=your_gmail_app_password
```

⚠️ Gmail App Password is required (not your Gmail login password).

---

# ▶️ Running the Application

Activate virtual environment:

```bash
venv\Scripts\activate
```

Run Flask app:

```bash
python app.py
```

Visit:

👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

# 🧪 Using the Application

### ✔ Upload Image

Choose a leaf image → click **Analyze Disease**

### ✔ View Prediction

It shows:

* Image preview
* Disease & plant name
* Confidence
* Symptoms, management & remedies
* Top-5 candidates

### ✔ Download PDF

Click **Download Report**

### ✔ Send Report

Enter:

* WhatsApp number
* Email
* Phone (SMS)

Click **Send Report**

### ✔ Explore Dashboard

Go to:

[http://127.0.0.1:5000/dashboard](http://127.0.0.1:5000/dashboard)

---

# 📊 Dashboard Insights

Charts generated using Plotly include:

| Chart                     | Description                           |
| ------------------------- | ------------------------------------- |
| **Disease Frequency**     | Shows most common detected diseases   |
| **Plant Distribution**    | Pie chart of plant categories         |
| **Prediction Confidence** | Histogram of model confidence         |
| **Severity Levels**       | Overview of detected disease severity |

---

# 🧬 AI Model Summary

* **Architecture:** EfficientNet-B0
* **Enhancement:** CBAM attention module
* **Training Dataset:** PlantVillage
* **Image Size:** 224×224
* **Accuracy:** ~99%
* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam

---

# 📚 Research Paper

The system corresponds to the research titled:

**“Performance Enhancement of Plant Disease Detection Using EfficientNet-B0 with CBAM Attention: A Comparative Deep Learning Study.”**

---

# 🧾 Prediction Storage

Every prediction is saved in:

```
predictions.json
```

Fields stored:

* plant
* disease
* confidence
* severity
* timestamp
* image path

---

# 🚀 Future Enhancements

* Mobile app integration
* Real-time camera detection
* Voice-based disease guidance
* Explainable AI (Grad-CAM heatmaps)
* On-device inference (TFLite optimization)

---

# 👨‍💻 Author
**Aakanksha Chaudhari**
**Sanika Deokule**
**Srushti Jadhav**
**Sakshi Rathod**

AIML Mini Project – Plant Disease Prediction System

---

# 📄 License

Open-source — free to use for academic & research purposes.

