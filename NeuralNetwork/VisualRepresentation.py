import matplotlib.pyplot as plt
import numpy as np

from NeuralNetwork.DataProcess import getMonthValues, getSpeiValues

def saveFig(plot, filepath, city_for_training, city_for_predicting):
    plt.savefig(filepath + f' - Model {city_for_training} applied to {city_for_predicting}')

def showSpeiData(xlsx, test_data, split, city_cluster_name, city_for_training, city_for_predicting, showImages):
    
    monthValues = getMonthValues(xlsx)
    speiValues, speiNormalizedValues =  getSpeiValues(xlsx)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(monthValues,speiValues,label='SPEI Original')
    plt.xlabel('Ano')
    plt.ylabel('SPEI')
    plt.title('SPEI Data - ' + city_for_training)
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(monthValues,speiNormalizedValues,label='Parcela de Treinamento')
    plt.xlabel('Ano')
    plt.ylabel('SPEI (Normalizado)')
    plt.plot(monthValues[split:],test_data,'k',label='Parcela de Teste')
    plt.legend()
    
    saveFig(plt, f'./Images/cluster {city_cluster_name}/model {city_for_training}/city {city_for_predicting}/SPEI Data', city_for_training, city_for_predicting)
    
def showSpeiTest(xlsx, test_data, split, city_cluster_name, city_for_training, city_for_predicting, showImages):
    
    monthValues = getMonthValues(xlsx)
    speiValues, speiNormalizedValues =  getSpeiValues(xlsx)

    positiveSpei = speiValues.copy()
    negativeSpei = speiValues.copy()

    y1positive=np.array(speiValues)>=0
    y1negative = np.array(speiValues)<=0

    plt.figure()
    plt.fill_between(monthValues, speiValues,y2=0,where=y1positive,
    color='green',alpha=0.5,interpolate=False, label='índices SPEI positivos')
    plt.fill_between(monthValues, speiValues,y2=0,where=y1negative,
    color='red',alpha=0.5,interpolate=False, label='índices SPEI negativos')
    plt.xlabel('Ano')
    plt.ylabel('SPEI')
    plt.title('SPEI Data - ' + city_for_training)
    plt.legend()
    if(showImages):
        plt.show()
    
    saveFig(plt, f'./Images/cluster {city_cluster_name}/model {city_for_training}/city {city_for_predicting}/SPEI Data', city_for_training, city_for_predicting)
    
def showPredictionResults(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, trainMonthForPredictedValues, testMonthForPredictedValues, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages):

    trueValues = np.append(trainDataTrueValues, testDataTrueValues)
    predictions = np.append(trainPredictValues, testPredictValues)

    reshapedMonth = np.append(trainMonthForPredictedValues, testMonthForPredictedValues)

    SpeiValues, SpeiNormalizedValues = getSpeiValues(xlsx)

    speiMaxValue = np.max(SpeiValues)
    speiMinValue = np.min(SpeiValues)

    trueValues_denormalized = (trueValues * (speiMaxValue - speiMinValue) + speiMinValue)
    predictions_denormalized = (predictions * (speiMaxValue - speiMinValue) + speiMinValue)

    plt.figure()
    plt.plot(reshapedMonth,trueValues_denormalized)
    plt.plot(reshapedMonth,predictions_denormalized)
    plt.axvline(trainMonthForPredictedValues[-1][-1], color='r')
    plt.legend(['Verdadeiros', 'Previstos'])
    plt.xlabel('Data')
    plt.ylabel('SPEI')
    plt.title('Valores verdadeiros e previstos para o final das séries.')
    if(showImages):
        plt.show()
    
    saveFig(plt, f'./Images/cluster {city_cluster_name}/model {city_for_training}/city {city_for_predicting}/Previsao', city_for_training, city_for_predicting)
    
def showPredictionsDistribution(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages):

    trueValues = np.append(trainDataTrueValues, testDataTrueValues)
    predictions = np.append(trainPredictValues, testPredictValues)

    SpeiValues, SpeiNormalizedValues = getSpeiValues(xlsx)

    speiMaxValue = np.max(SpeiValues)
    speiMinValue = np.min(SpeiValues)

    trueValues_denormalized = (trueValues * (speiMaxValue - speiMinValue) + speiMinValue)
    predictions_denormalized = (predictions * (speiMaxValue - speiMinValue) + speiMinValue)

    plt.figure()
    plt.scatter(x=trueValues_denormalized, y=predictions_denormalized, color=['white'], marker='^', edgecolors='black')
    plt.xlabel('SPEI Verdadeiros')
    plt.ylabel('SPEI Previstos')
    plt.axline((0, 0), slope=1)
    if(showImages):
        plt.show()
        
    saveFig(plt, f'./Images/cluster {city_cluster_name}/model {city_for_training}/city {city_for_predicting}/distribuiçãoDoSPEI', city_for_training, city_for_predicting)
