import unittest
import seaborn as sn
import pandas as pd
import finalDrafts.model
import dataPrep

import matplotlib.pyplot as plt
from dataExpander import *
import os
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from covnet import *
from learningStages import SplitLearner

in_shape = (96,96,3)


class MyTestCase(unittest.TestCase):

    def testSplitStage(self):
        dataPrep.stack.append(SplitStage((32, 32), 3))
        dataPrep.stack.append(FlipStage((32, 32)))
        dataPrep.stack.append(BlurStage((32, 32),20))
        dataPrep.stack.append(RotationStage((32, 32), 4))
        dataPrep.stack.append(NormalizeStage((32, 32), 9))
        dataPrep.load_all_dataset('test6')

    def testFinal(self):
        m =  finalDrafts.model.model('finalDrafts')

        dataPrep.stack.append(SplitStage((96, 96), 3))
        dataPrep.stack.append(RotationStage((96, 96), 4))
        dataPrep.stack.append(NormalizeStage((96, 96), 9))
        dataPrep.load_all_dataset('test',limit=10000)

        X = dataPrep.X
        pr = m.predict(X)
        y_pred = pr.numpy()
        Y_test = [i.argmax() for i in dataPrep.Y]
        conf = confusion_matrix(Y_test
                         ,pr)


        print('Precision: %.3f' % precision_score(Y_test, y_pred, average='micro'))
        print('Recall: %.3f' % recall_score(Y_test, y_pred, average='micro'))
        print('F1: %.3f' % f1_score(Y_test, y_pred, average='micro'))
        print('Accuracy: %.3f' % accuracy_score(Y_test, y_pred))

        df_cm = pd.DataFrame(conf, index=[i for i in range(1,9)],
                             columns=[i for i in range(1,9)])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.show()

    def testFromDeep(self):
        simpleCnn = load_model('Split_firstSplitStage_1.0')
        splittLearner = SplitLearner(simpleCnn, '__', baseLimit=10)

        splittLearner.limit = 100000
        splittLearner.path = 'Split____1.0'
        splittLearner.initData()
        splittLearner.fromDeepLearning()

    def testLearner(self):
        hiddenLayers = [
            Dense(100, activation='relu', name='firstHidden'),
            Dense(50, activation='relu', name='summarizer'),
            Dense(100, activation='relu', name='expander')
        ]

        # create the model to be trained by the learner
        simpleCnn = simpleOnLayerCov((96, 96, 3), 90, 3, (2, 2), 8, hiddenLayers, )

        splittLearner = SplitLearner(
            simpleCnn,
            'firstSplitStage',
            baseLimit=10000,
            baseEpoch=100
        )
        splittLearner.limit = 200

        splittLearner.verbose = 0
        splittLearner.OnBad = False

        # with this setting during the perform the epocs
        # will be randomly picked from a range of 10 and 10*1.5

        splittLearner.baseEpoch = 10
        splittLearner.epochIncrement = 1.5

        splittLearner.limit = 200

        splittLearner.perform()


if __name__ == '__main__':
    unittest.main()
