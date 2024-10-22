import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

with open("./NeuralNetwork/config.json") as arquivo:
    dados_json = json.load(arquivo)

parcelDataTrain= dados_json['parcelDataTrain']

def readXlsx(xlsx):
    df = pd.read_excel(xlsx)
    df.columns = df.columns.str.replace(' ', '')

    SpeiValues = df["Series1"].to_numpy()
    SpeiNormalizedValues = (SpeiValues-np.min(SpeiValues))/(np.max(SpeiValues)-np.min(SpeiValues))
    monthValues = df["Data"].to_numpy()

    return SpeiValues, SpeiNormalizedValues, monthValues

def splitSpeiData(xlsx):
    
    SpeiValues, SpeiNormalizedValues, monthValues = readXlsx(xlsx)

    speiTrainData, speiTestData, monthTrainData, monthTestData = train_test_split(SpeiNormalizedValues, monthValues, train_size=parcelDataTrain, shuffle=False)
    split = len(speiTrainData)
    
    return speiTrainData, speiTestData, monthTrainData, monthTestData, split