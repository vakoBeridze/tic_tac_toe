import numpy as np

from .train import create_model, load_images

model = create_model()
model.load_weights('model/weights.h5')

x, y = load_images("model_test")

x_test = np.asarray(x)
x_test = x_test.reshape(x_test.shape[0], 30, 30, 1)

prediction = model.predict_classes(x_test)

print(prediction)
print(y)
