import matplotlib.pyplot as plt
import numpy as np

from NeuralNetwork.DataProcess import readXlsx

def showSpeiData(xlsx, test_data, split, regionName,showImages,city):
    
    speiValues, speiNormalizedValues, monthValues = readXlsx(xlsx)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(monthValues,speiValues,label='SPEI Original')
    plt.xlabel('Ano')
    plt.ylabel('SPEI')
    plt.title('SPEI Data - ' + regionName)
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(monthValues,speiNormalizedValues,label='Parcela de Treinamento')
    plt.xlabel('Ano')
    plt.ylabel('SPEI (Normalizado)')
    plt.plot(monthValues[split:],test_data,'k',label='Parcela de Teste')
    plt.legend()
    if(city):
        plt.savefig('./Images/'+regionName+'/SPEI Data - Modelo '+ city)
    else:
        plt.savefig('./Images/'+regionName+'/SPEI Data')
    
def showSpeiTest(xlsx, test_data, split, regionName,showImages,city):
    
    speiValues, speiNormalizedValues, monthValues = readXlsx(xlsx)

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
    plt.title('SPEI Data - ' + regionName)
    plt.legend()
    if(showImages):
        plt.show()
    if(city):
        plt.savefig('./Images/'+regionName+'/SPEI Data - Modelo ' + city)
    else:
        plt.savefig('./Images/'+regionName+'/SPEI Data')
    
def showPredictionResults(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, trainMonthForPredictedValues, testMonthForPredictedValues, xlsx,regionName,showImages,city):

    trueValues = np.append(trainDataTrueValues, testDataTrueValues)
    predictions = np.append(trainPredictValues, testPredictValues)

    reshapedMonth = np.append(trainMonthForPredictedValues, testMonthForPredictedValues)

    speiValues, speiNormalizedValues, monthValues = readXlsx(xlsx)

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
    
    if(city):
        plt.savefig('./Images/'+regionName+'/Previsao - Modelo ' + city)
    else:
        plt.savefig('./Images/'+regionName+'/Previsao')
    
    
def showPredictionsDistribution(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, xlsx, regionName,showImages,city):

    trueValues = np.append(trainDataTrueValues, testDataTrueValues)
    predictions = np.append(trainPredictValues, testPredictValues)

    speiValues, speiNormalizedValues, monthValues = readXlsx(xlsx)

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
    if(city):
        plt.savefig('./Images/'+regionName+'/distribuiçãoDoSPEI - Modelo '+ city)
    else:
        plt.savefig('./Images/'+regionName+'/distribuiçãoDoSPEI')
