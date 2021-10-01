import tensorflow
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2


def predict_sample(model_path, sample_path):
    # load the sample image
    sample = cv2.imread(sample_path)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    sample = cv2.resize(sample, (224, 224))
    print(type(sample))

    # Generate a plot for the sample
    plt.imshow(sample)
    plt.show()
    print(sample.shape)

    samples_to_predict = [sample]
    samples_to_predict = np.array(samples_to_predict)
    print(samples_to_predict.shape)

    # Load the model
    model = load_model(model_path, compile=True)

    # Generate predictions for samples
    predictions = model.predict(samples_to_predict)
    print(predictions)

    # Generate arg maxes for predictions
    classes = np.argmax(predictions, axis=1)
    print(classes)
