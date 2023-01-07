import pickle

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, \
    confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle

import dataPrep
from covnet import *
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import *

from fractions import Fraction
from dataPrep import *


class LearningStage:
    def __init__(self,model,stageName,stageVersion='1.0',baseLimit=1000,baseEpoch=100,baseEqual=1.0):
        self.path = 'export/{}_{}'.format(stageName,stageVersion)

        self.epochIncrement = 1.5
        self.limitIncrement = 1.5

        self.baseLimit = baseLimit
        self.baseEpoch = baseEpoch

        self.model = model

        fract = Fraction(baseEqual)
        self.equals = [True for i in range(0,fract.denominator)]
        for i in range(0,fract.numerator-fract.numerator):
            self.equals[i] = False

        self.classToPrune=[]
        self.joinClass=[]
        self.epochs=100
        self.revDict={}
        self.verbose=0
        self.percentage=True
        self.OnBad=False

        self.stack = []

        self.limit = sys.maxsize
    def initData(self):
        dataPrep.stack = self.stack

        dataPrep.load_all_dataset(
            self.path,
            classToPrune=self.classToPrune,
            joinClass=self.joinClass,
            limit=self.limit,
            revDict=self.revDict,
            percentage=self.percentage
        )

        self.X = dataPrep.X
        self.Y = dataPrep.Y

    def visualizeFilter(self, layerName, maxFilters=9):
        filters, biases = self.model.get_layer(layerName).get_weights()
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        f, axarr = plt.subplots(maxFilters,maxFilters)
        l = maxFilters
        maxFilters = maxFilters**2

        k = 0
        for i in range(0, list(filters.shape)[-1]):
            for j in range(0, list(filters.shape)[-2]):
                f = filters[:, :, j,i]
                axarr[k//l,k%l].imshow(f, cmap='gray')
                maxFilters-=1
                k+=1
                if maxFilters == 0:
                    break
            if maxFilters == 0:
                break

        # show the figure
        plt.show()

    def perform(self):
        self.basePass()
        self.updateVariables()

    def updateVariables(self):
        pass

    def getFinalModel(self):
        pass

    def plotSome(self):
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.X[i])
        plt.show()

    def basePass(self):
        for i in range(0, 3):
            equal = random.choice(self.equals)
            scalingFactor = 1.0 if equal else 0.1

            self.percentage = equal
            limit = random.randint(self.baseLimit*scalingFactor, self.baseLimit*self.limitIncrement*scalingFactor)
            self.limit = limit

            self.initData()
            self.X,self.Y = shuffle(self.X, self.Y, random_state=0)

            ephocs = random.randint(self.baseEpoch * scalingFactor, self.baseEpoch * self.epochIncrement * scalingFactor)


            print("{}:\tep:{} lim:{} equ:{}".format(i, ephocs, limit, equal))

            if i % 1 == 0:
                self.model.save('export/{}'.format(self.path))
                covNetEvaluate(self.model,self.X,self.Y)

            covNetRetrain(self.model,self.X,self.Y,epochs=ephocs,verbose=self.verbose,OnBad=self.OnBad)

    def createIntermideLevel(self,model,layerName='expander'):
        X = np.asarray(self.X).astype(np.float32)

        _in = model.input
        _out = model.get_layer(layerName).output
        m = Model(_in, _out)

        self.X = m.predict(X)
        self.Y = np.array([np.argmax(i) for i in np.asarray(self.Y).astype(np.float32)])

        m.save('model')


    def fromDeepLearning(self):
        filename = 'randomTreeAll.sav'

        self.createIntermideLevel(self.model)
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.33, random_state=42)

        model = KNeighborsClassifier(8)

        classFiers = [
            KNeighborsClassifier(8),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            RandomForestClassifier(max_depth=8, n_estimators=10, max_features=1),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
        ]

        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            "Random Forest",
            "Naive Bayes",
            "QDA",
        ]

        print("Start fitting")

        scores = []

        for off, model in enumerate(classFiers):
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)

            plot_confusion_matrix(model, X_test, Y_test, cmap='GnBu')
            plt.show()
            print('Precision: %.3f' % precision_score(Y_test, y_pred, average='micro'))
            print('Recall: %.3f' % recall_score(Y_test, y_pred, average='micro'))
            print('F1: {}'.format(f1_score(Y_test, y_pred, average=None)))
            print('Accuracy: %.3f' % accuracy_score(Y_test, y_pred))
            scores.append(f1_score(Y_test, y_pred, average=None))

        for i in range(0, 8):
            best = 0
            bestF1 = 0.0
            for off, s in enumerate(scores):
                if s[i] > bestF1:
                    bestF1 = s[i]
                    best = off

            print("Class {} best is {} with {}".format(i + 1, names[best], scores[best]))

            pickle.dump(classFiers[best], open("oldClass/{}".format(i), 'wb'))

    def evaluate(self):
        self.initData()
        pr = self.model.predict(self.X)
        y_pred = [i.argmax() for i in pr]
        pr = y_pred
        Y_test = [i.argmax() for i in self.Y]
        print(Y_test)
        conf = confusion_matrix(Y_test, pr)

        print('F1: %.3f' % f1_score(Y_test, y_pred, average='micro'))

        df_cm = pd.DataFrame(conf, index=[i for i in range(0, 8)],columns=[i for i in range(0, 8)])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.show()


class SplitLearner(LearningStage):
    def __init__(self,model,stageName,stageVersion='1.0',baseLimit=1000,baseEpoch=100,baseEqual=1.0):
        LearningStage.__init__(self, model, "Split_"+ stageName, stageVersion, baseLimit, baseEpoch, baseEqual)
        inShape = list(model.input.shape)
        inShape = (inShape[1], inShape[2])

        self.stack.append(SplitStage(inShape,3))

        self.stack.append(FlipStage(inShape))
        self.stack.append(BlurStage(inShape,kern=3))

        self.stack.append(RotationStage(inShape,4))

        self.stack.append(NormalizeStage(inShape))

        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self.callback = EarlyStopping(monitor='val_loss',
                                                 patience=3,
                                                 restore_best_weights=True)

class AutoEncoderLearner(SplitLearner):
    def __init__(self,model,stageName,stageVersion='1.0',baseLimit=1000,baseEpoch=100,baseEqual=1.0):
        SplitLearner.__init__(self, model, "Split_"+ stageName, stageVersion, baseLimit, baseEpoch, baseEqual)
        self.second_stack = []


    def perform(self):

        SplitLearner.perform(self)





