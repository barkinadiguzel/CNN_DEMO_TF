import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("cnn_model.h5")

# Dummy image for prediction
sample = np.random.rand(1, 28, 28, 1)
prediction = model.predict(sample)

print("Predicted class:", np.argmax(prediction))
