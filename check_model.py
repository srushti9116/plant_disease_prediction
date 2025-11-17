from tensorflow.keras.models import load_model

MODEL_PATH = "model/plant_disease_model.h5"

model = load_model(MODEL_PATH, compile=False)
print("Model input shape :", model.input_shape)
print("Model output shape:", model.output_shape)
