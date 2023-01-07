import os
import random
import shutil
from shutil import copyfile
import tqdm

def getSub(s,number):
    files = os.listdir('training_data_final/Species{}'.format(s))
    random.shuffle(files)
    return files[:number]

{(7, 4): 180,
 (3, 5): 145,
 (3, 1): 48,
 (7, 6): 74,
 (1, 5): 37,
 (7, 2): 165,
 (4, 3): 135,
 (5, 4): 163,
 (2, 7): 146,
 (5, 6): 87,
 (3, 4): 152,
 (2, 3): 167,
 (2, 4): 162,
 (4, 2): 123,
 (3, 7): 148,
 (7, 1): 56,
 (7, 3): 174,
 (5, 3): 159,
 (5, 7): 174,
 (6, 5): 70,
 (6, 7): 51,
 (4, 7): 130,
 (4, 6): 47,
 (6, 3): 62,
 (7, 5): 172,
 (1, 3): 41,
 (5, 2): 179,
 (3, 6): 69,
 (5, 1): 62,
 (3, 2): 135,
 (6, 4): 45,
 (2, 5): 161,
 (4, 1): 49,
 (4, 5): 131,
 (1, 4): 40,
 (2, 1): 55,
 (1, 7): 35,
 (1, 2): 43,
 (6, 2): 45,
 (2, 6): 53,
 (6, 1): 11,
 (1, 6): 18}

toRecount = {
        4:87,
        7:88,
        5:87,
        2:92,
        8:82,
        6:40
}

newNonClass = {}
for k in toRecount:
    newNonClass[k] = getSub(k,toRecount[k])

if not os.path.exists('repartedDataset/Species1'):
    os.mkdir('repartedDataset/Species1')

counter = 0
for k in tqdm.tqdm(newNonClass,desc='Transforming original dataset splitting'):
    for f in newNonClass[k]:
        shutil.copyfile('training_data_final/Species{}/{}'.format(k,f),'Species1/{}.jpg'.format(counter))
        counter += 1
