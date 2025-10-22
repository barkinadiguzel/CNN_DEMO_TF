import numpy as np

def normalize_images(x):
    return x.astype('float32') / 255.0

def summarize_model(model):
    print("\n🧠 Model Summary:")
    model.summary()
