import uuid
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
import tensorflow as tf
import numpy as np


def covNetRetrain(model,X,Y,epochs,verbose,OnBad):
    """This function will retrain a given  covnet model

    Args:
        model (Model): the model to retrain
        X (numpy array): the input data 
        Y (numpy array): the input data
        epochs (integer): the number of ephocs
        verbose (flag): if you want to put the keras output
        OnBad (flag): if you want to retrain on missed data
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)

    if OnBad:
        predicted = tf.argmax(np.array(model.predict(X_train)),axis=-1)
        real  = tf.argmax(y_train,axis=-1)

        new_x = []
        new_y = []

        for off,_ in enumerate(predicted):
            if _ != real[off]:
                new_x.append(X_train[off])
                new_y.append(y_train[off])

    model.fit(X_train, y_train, epochs=epochs, verbose=verbose)


def covNetEvaluate(model, X,Y):
    """This function evaluate a given model

    Args:
        model (Model): the model to evaluate
        X (numpy array): the input data 
        Y (numpy array): the input data
    """
    _, X_test,_, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    model.evaluate(X_test, y_test)

def mergeModels(models,commonInput):
    """This function concatenate different models

    Args:
        models (Array ): a list of models
        commonInput (A keras input layer): the input layer

    Returns:
        Model : the merged model
    """
    toMerge = []
    for model in models:
        toMerge.append(model(commonInput))


    conc = Concatenate(toMerge, name="mergedLayer{}".format(uuid.uuid4()))
    return conc


def simpleOnLayerCov(inputShape,numBaseFilter,layers,filterDimension,K,hiddenLayers):
    """Generate a simple Cnn model

    Args:
        inputShape (shape): the shape of input
        numBaseFilter (number): the basic number to filter to use
        layers (integer): the number of cnn layers that the model must have
        filterDimension (shape): the dimension of filter
        K (integer): the number of classes
        hiddenLayers (list): the hidden layers

    Returns:
        Model: a Cnn network
    """
    model = Sequential()

    model.add(Conv2D(numBaseFilter, filterDimension, activation='relu', padding='same', input_shape=inputShape))
    model.add(BatchNormalization())

    for i in range(0,layers):
        model.add(Conv2D(numBaseFilter, filterDimension, activation='relu', padding='same', input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        numBaseFilter *=2

    model.add(Flatten())
    model.add(Dropout(0.2))

    # Hidden layer
    for i in hiddenLayers:
        model.add(i)
        model.add(Dropout(0.2))

    model.add(Dense(K, activation='softmax'))

    # model description
    model.summary()
    return model








