import math
import random
import re

import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import copy
import os
import tqdm


from threading import Thread
import multiprocessing as mp

def pipeExternalConcStep(instance,i,j):
    instance.startPipelineForInput(i,j)

def pipeExternalConcStepAutoEncoder(instance,i,j,k):
    instance.startPipelineForInput(i,j,k)

def rotateExternalConcStep(instance,i):
    instance.rotate(i)

def splitExternalConcStep(instance,i,j):
    instance.split(i,j)

def pruneExternalConcStep(instance,i):
    instance.rotate(i)


def doParalllel(myFunction,inputs,instance):

        process = []

        for i in inputs:
            i.insert(0, instance)
            i = tuple(i)
            new_t = Thread(target=myFunction, args=i)
            process.append(new_t)
            new_t.start()

        for p in process:
            p.join()




class PreprocessingStage():
    def __init__(self,X):
        self.data = X
        self.Y = None
        self.isMultiOutput = False

    def perform(self):
        pass
class ScalingStage(PreprocessingStage):
    def __init__(self,X,sizes):
        self.sizes = sizes
        PreprocessingStage.__init__(self,X)

    def perform(self,toPerform=True):
        if toPerform:
            self.Y = cv2.resize(self.X,self.sizes)
        else:
            self.Y = self.X

class NormalizeStage(ScalingStage):
    def __init__(self,sizes,X=None):
        ScalingStage.__init__(self,X,sizes)

    def perform(self):
        #self.Y = cv2.fastNlMeansDenoisingColored(self.X, None, 20, 20, 7, 21)
        self.Y = self.X

        self.X,self.Y = self.Y,None
        ScalingStage.perform(self)

class BlurStage(ScalingStage):
    def __init__(self,sizes,kern,X=None):
        ScalingStage.__init__(self,X,sizes)
        self.Y = []
        self.isMultiOutput = True
        self.kern = kern

    def perform(self):

        self.Y.append(self.X)
        for i in range(2,self.kern):
            self.Y.append(cv2.blur(self.X,(i, i)))

class FlipStage(ScalingStage):
    def __init__(self,sizes,X=None):
        ScalingStage.__init__(self,X,sizes)
        self.Y = []
        self.isMultiOutput = True

    def perform(self):

        self.Y.append(self.X)
        flip_v = cv2.flip(self.X,0)
        flip_h = cv2.flip(self.X,1)

        flip_v_h = cv2.flip(flip_v, 1)

        self.Y.append(flip_h)
        self.Y.append(flip_v_h)
        self.Y.append(flip_v)




class SplitStage(ScalingStage):
    def __init__(self,sizes,splits,X=None):
        ScalingStage.__init__(self,X,sizes)
        self.Y = []
        self.splits = splits
        self.isMultiOutput = True

    def perform(self):

        self.imgheight = self.X.shape[0]
        self.imgwidth = self.X.shape[1]

        self.M = self.imgheight // self.splits
        self.N = self.imgwidth // self.splits

        self.Y.append(self.X)

        inputs = []
        for y in range(0, self.imgheight, self.M):
            for x in range(0, self.imgwidth, self.N):
                inputs.append([x,y])


        myFunction = splitExternalConcStep
        doParalllel(myFunction,inputs,self)

    def split(self,x,y):
        tiles = self.X[y:y + self.M, x:x + self.N]
        self.Y.append(tiles)






class RotationStage(ScalingStage):
    def __init__(self,sizes,angleStep=10,X=None):
        ScalingStage.__init__(self,X,sizes)
        self.angleStep = angleStep
        self.Y = []
        self.isMultiOutput = True

    def perform(self):
        (self.h, self.w) = self.X.shape[:2]
        (self.cX, self.cY) = (self.w // 2, self.h // 2)

        self.Y.append(self.X)

        inputs = [ [i] for i in range(0,self.angleStep)]
        myFunction = rotateExternalConcStep

        doParalllel(myFunction,inputs,self)


    def rotate(self,i):
        angle = 360 / self.angleStep * i
        M = cv2.getRotationMatrix2D((self.cX, self.cY), angle, 1.0)
        rotated = cv2.warpAffine(self.X, M, (self.w, self.h))

        self.Y.append(rotated)


class SplittedDataset:
    def __init__(self,X,Y,ratio = 0.33):
        self.X = X
        self.Y = Y
        self.ratio = ratio

    def split(self):
        return  train_test_split(self.X, self.Y, test_size=self.ratio)
class DatasetPreparer:
    def __init__(self,sizes,path):
        self.X = []
        self.Y = []
        self.dir = path
        self.sizes = sizes

        self.startDataset = []

        self.pipeline:[PreprocessingStage] = []

    def addToPipeline(self,stage:PreprocessingStage):
        self.pipeline.append(stage)

    def loadStartDataset(self,filenames,Ys):
        for off,name in enumerate(filenames):
            img = cv2.imread(name)
            if img is None:
                continue
            self.startDataset.append((img,Ys[off]))

        pass

    def loadInitialDatasetPreexpanded(self,limit=2000,percentage=None):
        self.X = []
        self.Y = []

        dirs = os.listdir(self.dir)
        random.shuffle(dirs)
        files = 0
        Ys = []

        #zone dedicated for balance dataset
        classes = list(set([i.split('-')[0] for i in dirs]))
        toUse = {}

        for i in classes:
            toAppend = []
            for j in dirs:
                if i in j:
                    toAppend.append(j)
            toUse[i] = toAppend

        balanced = 0

        for k in range(0,len(dirs)//len(toUse)):
            for j in toUse:
                if balanced>=len(toUse[j]):
                    continue
                dir = toUse[j][balanced]
                files +=1
                path = '{}/{}'.format(self.dir,dir)
                if os.path.isdir(path):
                    continue

                if files == limit:
                    break
                image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

                self.X.append(image)

                self.Y.append(np.fromstring(dir.split('-')[0].replace('[','').replace(']',''),dtype=float,sep=' '))
            if files == limit:
                break
            balanced +=1

    def loadInitialDatasetFromSub(self,sub,revDict):
        dirs = os.listdir('{}'.format(sub))
        files = []
        Ys = []

        for dir in dirs:
            if os.path.isfile('{}/{}'.format(sub,dir)):
                continue

            f = os.listdir('{}/{}'.format(sub,dir))
            for k in f:
                if k.startswith(","):
                    continue
                files.append('{}/{}/{}'.format(sub,dir, k))
                Ys.append(revDict[dir])

        self.loadStartDataset(files,Ys)


    def executePipeline(self):
        myFunction = self.startPipelineForInput
        inputs = tqdm.tqdm(
            [[i[0],i[1]] for i in self.startDataset],
                                  desc='Transforming original dataset')

        doParalllel(pipeExternalConcStep,inputs,self)






    def getSplittedDataset(self,ratio=0.33):
        return SplittedDataset(self.X,self.Y,ratio)

    def exportDataset(self,path='dataset'):
        if not os.path.exists(path):
            os.mkdir(path)

        for off,img in enumerate(self.X):
            cv2.imwrite('{}/{}-image-{}.png'.format(path,self.Y[off],off),img)

    def startPipelineForInput(self,input,Y=None,i=0):

        stage = copy.deepcopy(self.pipeline[i])
        stage.X = input
        stage.perform()


        if stage.isMultiOutput:
            if i == len(self.pipeline) - 1:
                for j in stage.Y:
                    self.X.append(j)
                    self.Y.append(Y)
                return
            else:
                for j in stage.Y:
                    self.startPipelineForInput(i=i+1,input=j,Y=Y)


        else:
            if i == len(self.pipeline) - 1:
                self.X.append(stage.Y)
                self.Y.append(Y)

                return
            else:
                self.startPipeline(i+1,stage.Y,Y=Y)


    def getEquivalence(self,registered,toEval):
        new_ = [
                   i for i in registered if toEval in i
               ]
        return list(set([toEval] +([] if len(new_) == 0 else new_[0])))


    def pruneClass(self,classesToPrune = [],joinClass=[],revDict={}):
        if len(classesToPrune) == 0 and len(joinClass) == 0:
            return

        remainedX = []
        remainedY = []




        isClassRegistered = lambda cls,registered,joinClass: True in [tuple(self.getEquivalence(joinClass,i)) in registered for i in self.getEquivalence(joinClass,cls)]

        for off,Y in tqdm.tqdm(enumerate(self.Y),desc='Start pruning'):
            i = Y
            cls = np.unravel_index(i.argmax(), i.shape)[0] + 1
            if cls in classesToPrune:
                continue

            key = set(self.getEquivalence(joinClass,cls))
            if not isClassRegistered(cls,revDict,joinClass):
                revDict[tuple(key)] = len(revDict.keys())+1
            remainedX.append(self.X[off])
            remainedY.append(revDict[tuple(key)])


        print(revDict)
        revDict_second = {}
        kk = len(revDict.keys())
        for k in revDict:
            i = revDict[k]
            label = np.zeros((1, kk))
            label[0][i-1] = 1
            revDict_second[i] = label

        for off,i in enumerate(remainedY):
            remainedY[off] = revDict_second[i]

        self.X = remainedX
        self.Y = remainedY


class AutoEncoderExpander(DatasetPreparer):
    def __init__(self,path,endDimension):
        DatasetPreparer.__init__(self,sizes=endDimension,path=path)
        self.endDimension = endDimension
        self.Y = {}


    def exportDataset(self,path='dataset'):
        if not os.path.exists(path):
            os.mkdir(path)


        for off,img in enumerate(self.X):
            for off2,img2 in enumerate(self.Y[off]):
                cv2.imwrite('{}/export-{}-{}.png'.format(path,off2,off),img2)
            cv2.imwrite('{}/image-{}-{}.png'.format(path,off2,off),img)



    def executePipeline(self):
        myFunction = self.startPipelineForInput
        inputs = tqdm.tqdm(
            [[i[0],i[1],off] for off,i in enumerate(self.startDataset)],
                                  desc='Transforming original dataset using autoencoding one')

        doParalllel(pipeExternalConcStepAutoEncoder,inputs,self)
        self.Y = [self.Y[i] for i in self.Y]



    def startPipelineForInput(self,input,Y=None,i=0, whereToSave = 0):

        stage = copy.deepcopy(self.pipeline[i])
        stage.X = input
        stage.perform()

        if i==0:
            self.X.append(input)
            self.Y[whereToSave] = []


        if stage.isMultiOutput:
            if i == len(self.pipeline) - 1:
                for j in stage.Y:
                    self.Y[whereToSave].append(Y)
                return
            else:
                for j in stage.Y:
                    self.startPipelineForInput(i=i+1,input=j,Y=Y,whereToSave=whereToSave)


        else:
            if i == len(self.pipeline) - 1:
                self.Y[whereToSave].append(stage.Y)

                return
            else:
                self.startPipeline(i+1,stage.Y,Y=Y,whereToSave=whereToSave)


    def loadInitialDatasetPreexpanded(self,limit=2000,percentage=None):
        dirs = os.listdir(self.dir)
        random.shuffle(dirs)
        files = 0
        Ys = []

        for dir in dirs:
            if dir.split("-")[0] == "subscale":
                continue

            path = '{}/{}'.format(self.dir,dir)
            if os.path.isdir(path):
                continue

            files +=1
            if files == limit:
                break
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


            subElements = os.listdir(self.dir)
            reg = re.compile('subscale-{}-.*')
            subElements = reg.findall(subElements)

            for i in subElements:
                self.X.append(image)
                self.Y.append(
                    np.array(cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)).resize(-1)
                )





