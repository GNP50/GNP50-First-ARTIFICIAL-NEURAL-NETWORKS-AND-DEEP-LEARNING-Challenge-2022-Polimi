from sklearn.metrics import *
import dataPrep
from dataPrep import *
import pickle
from covnet import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



def trainAndPlot(model):
    print("Start fitting")
    Y_train = [np.unravel_index(i.argmax(), i.shape)[0]+1 for i in dataPrep.y_train]
    Y_test = [np.unravel_index(i.argmax(), i.shape)[0]+1 for i in dataPrep.y_test]
    model.fit(dataPrep.X_train.reshape(-1,96*96*3), Y_train)
    y_pred = model.predict(dataPrep.X_test.reshape(-1,96*96*3))

    plot_confusion_matrix(model, dataPrep.X_test.reshape(-1,96*96*3), Y_test, cmap='GnBu')
    plt.show()
    print('Precision: %.3f' % precision_score(Y_test, y_pred,average='micro'))
    print('Recall: %.3f' % recall_score(Y_test, y_pred,average='micro'))
    print('F1: %.3f' % f1_score(Y_test, y_pred,average='micro'))
    print('Accuracy: %.3f' % accuracy_score(Y_test, y_pred))

def allRandomTree(classToPrune=[],joinClass=[],revDict={}):
    filename = 'randomTreeAll.sav'

    load_all_dataset(classToPrune=classToPrune,joinClass=joinClass,revDict=revDict)
    model = RandomForestClassifier()
    trainAndPlot(model)

    pickle.dump(model, open(filename, 'wb'))






