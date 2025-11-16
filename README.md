# 🌱 AgroVision AI – Plant Disease Prediction System

AgroVision AI is a mini-project that uses a **Convolutional Neural Network (CNN)** to detect plant diseases from leaf images and provide:

- Disease name and affected plant  
- Model confidence and severity level  
- Description, symptoms and management tips  
- Example remedies / tonics with links to e-commerce sites (Amazon / Flipkart)

The project is built using **Python, TensorFlow/Keras, and Flask** with a modern UI.

---

## 🧠 Tech Stack

- **Backend:** Python, Flask  
- **ML / DL:** TensorFlow, Keras, CNN-based image classifier  
- **Frontend:** HTML5, CSS3, JavaScript  
- **Other:** NumPy, Pillow, JSON  

---

## 📁 Project Structure

```text
plant_disease_app/
├─ app.py                    # Main Flask application
├─ model/
│  ├─ class_indices.json     # Class index to label mapping
│  └─ plant_disease_model.h5 # Trained CNN model (NOT in repo – download separately)
├─ templates/
│  ├─ index.html             # Home page (upload leaf image)
│  ├─ result.html            # Prediction result page
│  ├─ how_it_works.html      # Explanation of workflow
│  └─ about.html             # Project details page
├─ static/
│  ├─ css/
│  │  └─ style.css           # UI styling (glassmorphism, animations, etc.)
│  ├─ remedies/              # Example product images for remedies
│  └─ uploads/               # Uploaded leaf images (runtime)
├─ .gitignore
└─ README.md
