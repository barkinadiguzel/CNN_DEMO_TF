import numpy as np
from models.cnn_model import build_cnn_model

# Generate dummy data (1000 grayscale 28x28 images)
x_train = np.random.rand(1000, 28, 28, 1)
y_train = np.random.randint(0, 10, 1000)

x_test = np.random.rand(200, 28, 28, 1)
y_test = np.random.randint(0, 10, 200)

# Build and train the CNN
model = build_cnn_model()
model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test))

# Save model
model.save("cnn_model.h5")
print("✅ Model trained and saved as cnn_model.h5")
