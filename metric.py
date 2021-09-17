import config


import pickle
import math
import os
import datetime


metricFile = None

def start_prediction_eval():
    global metricFile
    targets = []
    outputs = []

    # Opening the file now to write the metrics score
    os.makedirs(os.path.dirname(config.METRICS_PATH.format(config.DATASET)), exist_ok=True)
    metricFile = open(config.METRICS_PATH.format(config.DATASET), 'a+')
    metricFile.write("\n"+str(datetime.datetime.now())+"\n")
    # Writing hyperparameters to file
    line = "epochs: {}, lr: {}, weight_decay: {}, momentum: {}, k: {}, cluster_nos: {} walk_length: {}, window_size: {}, walk_nos: {}\n".format(
        config.EPOCHS, config.LEARNING_RATE, config.WEIGHT_DECAY, config.MOMENTUM, config.TOP_K, config.CLUSTERS_NUM, config.WALK_LENGTH, config.WINDOW_SIZE, config.NUMBER_WALK
    )
    metricFile.write(line)

    with open(config.PREDICTIONS_PATH.format(config.DATASET), 'rb') as f:
        tempTarget = pickle.load(f)
        outputs    = pickle.load(f)

    targets = []
    for eachRow in tempTarget:
        eachRow = [i for i,e in enumerate(eachRow) if e == 1]
        targets.append(eachRow)

    topK = [1,3,5]
    precision(topK, targets, outputs)
    nDCG(topK, targets, outputs)
    percentage(targets, outputs) #Only to be called when you feel like ;)
    metricFile.close()
    print('Evaluation is finished. Evaluation results are at ',config.METRICS_PATH.format(config.DATASET))




def precision(topK, targets, outputs):
    totalTestNos = len(targets)
    for eachK in topK:
        k = eachK
        precision = 0

        # Traversing the Lists
        for index in range(0, totalTestNos):
            eachOutput = outputs[index][0:k]
            count      = 0
            for eachElem in eachOutput:
                if eachElem in targets[index]:
                    count = count + 1
            precision = precision + count/k

        precision = precision/totalTestNos
        line      = 'Precision@{}: {}\n'.format(k, precision)
        print(line)
        metricFile.write(line)


def nDCG(topK, targets, outputs):
    totalTestNos = len(targets)
    for eachK in topK:
        idCg   = 0
        dCGList = []
        for dataIndex in range(0, len(targets)): #eachRow in outputs:
            dCG       = 0
            eachRow   = outputs[dataIndex][0:eachK]
            targetRow = targets[dataIndex]
            for index in range(0, len(eachRow)):
                if eachRow[index] in targetRow:
                    dCG = dCG + 1/math.log(1+index+1)
            dCGList.append(dCG)
            if dCG > idCg:
                idCg = dCG
        if idCg == 0:
            idCg = 1
        dCGList = [x / idCg for x in dCGList]

        line    = 'nDCG@{}: {}\n'.format(eachK, sum(dCGList)/totalTestNos)
        print(line)
        metricFile.write(line)

# This function calculates the percentage of real labels in predicted label vectors   
# This is not included in the original paper.
def percentage(targets, outputs):
    percentageScore = 0
    for index in range(0, len(targets)):
        targetRow = targets[index]
        outputRow = outputs[index]

        count     = 0
        for eachElem in outputRow:
            if eachElem in targetRow:
                count += 1
        percentageScore += count/len(targetRow)

    finalPercentageScore = percentageScore/len(targets)
    line = 'Percentage Score: {}\n'.format(finalPercentageScore)
    print(line)
    metricFile.write(line)