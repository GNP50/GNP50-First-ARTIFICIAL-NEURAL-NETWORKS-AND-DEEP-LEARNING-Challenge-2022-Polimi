import os

import numpy as np
import tensorflow as tf


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'model'))


    def predict(self, X):
        #finalImage = []
        #for image in X:
        #    finalImage.append(cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21))
        #X = np.array(finalImage).reshape(-1,96,96,3)

        X = np.array(X).reshape(-1,96,96,3)
        out = self.model.predict(X)

        out =  np.array(out)
        out = tf.argmax(out, axis=-1)

        return out