import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

with open("./NeuralNetwork/config.json") as arquivo:
    dados_json = json.load(arquivo)

parcelDataTrain  = dados_json['parcelDataTrain']
predictionPoints = dados_json['predictionPoints']

def readXlsx(xlsx):
    df         = pd.read_excel(xlsx)
    df.columns = df.columns.str.replace(' ', '')

    SpeiValues           = df["Series1"].to_numpy()
    SpeiNormalizedValues = (SpeiValues-np.min(SpeiValues))/(np.max(SpeiValues)-np.min(SpeiValues))
    monthValues          = df["Data"].to_numpy()

    return SpeiValues, SpeiNormalizedValues, monthValues

def splitSpeiData(xlsx):
    
    SpeiValues, SpeiNormalizedValues, monthValues = readXlsx(xlsx)
    
    DATA_TYPES_LIST = ['Train', 'Test']
    SPEI_dict       = dict.fromkeys(DATA_TYPES_LIST)
    months_dict     = dict.fromkeys(DATA_TYPES_LIST)
    
    SPEI_dict['Train'], SPEI_dict['Test'], months_dict['Train'], months_dict['Test']  = train_test_split(SpeiNormalizedValues, monthValues, train_size=parcelDataTrain, shuffle=False)
    split = len(SPEI_dict['Train'])
    
    return SPEI_dict, months_dict, split

def cria_IN_OUT(data, janela):
    OUT_indices = np.arange(janela, len(data), janela)
    OUT         = data[OUT_indices]
    lin_x       = len(OUT)
    IN          = data[range(janela*lin_x)]
   
    IN          = np.reshape(IN, (lin_x, janela, 1))

    OUT_final   = IN[:,-predictionPoints:,0]
    IN_final    = IN[:,:-predictionPoints,:]
    return IN_final, OUT_final