import os.path

import tensorflow

from dataExpander import *

s = (slice(0, 96), slice(0, 96))


X,Y = None,None
stack = []
realod=False

def load_all_WithPreparer(folder,baseDir,limit,datasetPreparer=None):
    LocalRealod =  checkIfIsToRealod(baseDir,folder)


    _revDict = {}
    for i in range(0, 8):
        label = np.zeros((1, 8))
        label[0][i] = 1
        _revDict['Species{}'.format(i + 1)] = label

    if LocalRealod:
        datasetPreparer.loadInitialDatasetFromSub('training_data_final', revDict=_revDict)
        for step in stack:
            datasetPreparer.addToPipeline(step)
        datasetPreparer.executePipeline()
        datasetPreparer.exportDataset(folder)

    datasetPreparer.loadInitialDatasetPreexpanded(limit=limit)

def percentage_dataset(percentage,X,Y):
    if percentage:
        datasetPreparer = np.argmax(Y, axis=-1)
        counts = [
            np.count_nonzero(
                datasetPreparer == i
            )

            for i in range(0, 8)
        ]

        m = min(counts)
        new_y = []
        new_x = []

        for i in range(0, 8):
            actualLengh = 0
            for off, ele in enumerate(datasetPreparer):
                if ele == i:
                    actualLengh += 1
                    new_x.append(X[off])
                    new_y.append(Y[off])
                if actualLengh == m:
                    break
        X = np.array(new_x)
        Y = np.array(new_y)

    return (X,Y)

def checkIfIsToRealod(baseDir,folder):
    if not os.path.exists(baseDir):
        os.mkdir(baseDir)
    LocalRealod = False
    if not os.path.exists(folder) or realod:
        os.mkdir(folder)
        LocalRealod = True

    print(LocalRealod)
    return LocalRealod

def load_all_dataset(folder,classToPrune=[],limit=100000,joinClass=[],revDict={},percentage=True,datasetPreparer=None):

    global X,Y,X_train, X_test, y_train, y_test,stack,realod
    baseDir = "../datasets/"
    folder = "../datasets/" + folder.split("/")[-1]

    try:
        if datasetPreparer is None:
            datasetPreparer = DatasetPreparer(sizes=s, path=folder)

        load_all_WithPreparer(folder,baseDir,limit,datasetPreparer=datasetPreparer)

        datasetPreparer.pruneClass(classToPrune,joinClass=joinClass,revDict=revDict)
        clsNum = list(datasetPreparer .Y[0].shape)[-1]
        X = np.array(datasetPreparer .X )
        Y = np.array(datasetPreparer .Y,dtype=object).reshape(-1, clsNum)

        X,Y = percentage_dataset(percentage,X,Y)

    except Exception as inst:
        print(inst.args)
        print(inst)
        if os.path.exists(folder):
            os.rmdir(folder)


def load_all_withoutclass(folder,limit=100000, percentage=True,datasetPreparer=None):
    global X, Y, X_train, X_test, y_train, y_test, stack, realod
    baseDir = "../datasets/"
    folder = "../datasets/" + folder.split("/")[-1]

    try:

        load_all_WithPreparer(folder,baseDir,limit, datasetPreparer=datasetPreparer)
        datasetPreparer.loadInitialDatasetPreexpanded()

        X = np.array(datasetPreparer.X)
        Y = np.array(datasetPreparer.Y)

        X, Y = percentage_dataset(percentage, X, Y)

    except Exception as inst:
        print(inst.args)
        print(inst)
        if os.path.exists(folder):
            os.rmdir(folder)


