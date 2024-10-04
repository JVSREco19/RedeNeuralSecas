import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

from NeuralNetwork.DataProcess import splitSpeiData
from NeuralNetwork.VisualRepresentation import showPredictionResults, showPredictionsDistribution, showSpeiData, showSpeiTest
from NeuralNetwork.Metrics import getError

metricsCompendium = {}

# Abra o arquivo JSON
with open("NeuralNetwork\config.json") as arquivo:
    dados_json = json.load(arquivo)

totalPoints= dados_json['totalPoints']
predictionPoints= dados_json['predictionPoints']
numberOfEpochs = dados_json['numberOfEpochs']
hiddenUnits = dados_json['hiddenUnits']

def createNeuralNetwork(hidden_units, dense_units, input_shape, activation):
    model = tf.keras.Sequential()   
    model.add(tf.keras.layers.LSTM(hidden_units,input_shape=input_shape,activation=activation[0]))
    model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
    model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
    model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def trainNeuralNetwork(trainDataForPrediction, trainDataTrueValues,showImages,regionName):

    model = createNeuralNetwork( hidden_units= hiddenUnits, dense_units=predictionPoints, 
                                input_shape=(totalPoints-predictionPoints,1), activation=['relu','sigmoid'])
    print(model.summary())

    #treina a rede e mostra o gráfico do loss
    history=model.fit(trainDataForPrediction, trainDataTrueValues, epochs=numberOfEpochs, batch_size=1, verbose=0)
    plt.figure()
    plt.plot(history.history['loss'],'k')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend(['loss'])
    if(showImages):
        plt.show()
    plt.savefig('./Images/'+regionName+'/MSE')
    return model

def cria_IN_OUT(data, janela):
    OUT_indices = np.arange(janela, len(data), janela)
    OUT = data[OUT_indices]
    lin_x = len(OUT)
    IN = data[range(janela*lin_x)]
   
    IN = np.reshape(IN, (lin_x, janela, 1))

    OUT_final = IN[:,-predictionPoints:,0]
    IN_final = IN[:,:-predictionPoints,:]
    # print(OUT_final)
    # print('-------------------------\n')
    # print(IN_final)
    return IN_final, OUT_final

def FitNeuralNetwork(xlsx, regionName,showImages):

        #[0] = lista de dados do SPEI referentes à parcela de treinamento (80%)
        #[1] = lista de dados do SPEI referentes à parcela de teste (20%)
        #[2] = lista de datas referentes à parcela de treinamento (80%)
        #[3] = lista de datas referentes à parcela de teste (20%)
        #[4] = valor inteiro da posição que o dataset foi splitado
    trainData, testData, monthTrainData, monthTestData, split = splitSpeiData(xlsx)

        #[0] = Dataset que contem a parcela de dados que será utilizada para alimentar a predição da rede(treinamento)
        #[1] = Dataset que contem a parcela de dados que será utilizada para validar se as predições da rede estão corretas(treinamento)
    trainDataForPrediction, trainDataTrueValues = cria_IN_OUT(trainData, totalPoints)

        #[0] = Dataset que contem a parcela de dados que será utilizada para alimentar a predição da rede(teste)
        #[1] = Dataset que contem a parcela de dados que será utilizada para validar se as predições da rede estão corretas(teste)
    testDataForPrediction, testDataTrueValues = cria_IN_OUT(testData, totalPoints)

        #[0] = Dataset que contem a parcela dos meses nos quais os SPEIs serão utilizados para alimentar a predição da rede(treinamento)
        #[1] = Dataset que contem a parcela dos meses nos quais os SPEIs serão preditos(treinamento)
    trainMonthsForPrediction, trainMonthForPredictedValues = cria_IN_OUT(monthTrainData, totalPoints)

        #[0] = Dataset que contem a parcela dos meses nos quais os SPEIs serão utilizados para alimentar a predição da rede(teste)
        #[1] = Dataset que contem a parcela dos meses nos quais os SPEIs serão preditos(teste)
    testMonthsForPrediction, testMonthForPredictedValues = cria_IN_OUT(monthTestData, totalPoints)

    model = trainNeuralNetwork(trainDataForPrediction, trainDataTrueValues,showImages,regionName)

        #faz previsões e calcula os erros
    trainPredictValues = model.predict(trainDataForPrediction)
    testPredictValues = model.predict(testDataForPrediction)

    trainErrors = getError(trainDataTrueValues, trainPredictValues)
    testErrors = getError(testDataTrueValues, testPredictValues)
    if regionName not in metricsCompendium:
        metricsCompendium[regionName] = {}
    if regionName not in metricsCompendium[regionName]:
        metricsCompendium[regionName][regionName] = {"trainErrors": [], "testErrors": []}
        
    metricsCompendium[regionName][regionName]["trainErrors"].append(trainErrors)
    metricsCompendium[regionName][regionName]["testErrors"].append(testErrors)

    print("--------------Result for " + regionName +"---------------")
    print("---------------------Train-----------------------")
    print(trainErrors)

    print("---------------------Test------------------------")
    print(testErrors)

    showSpeiData(xlsx, testData, split, regionName,showImages,regionName)
    showSpeiTest(xlsx, testData, split, regionName,showImages,regionName)
    showPredictionResults(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, trainMonthForPredictedValues, testMonthForPredictedValues, xlsx,regionName,showImages,regionName)
    showPredictionsDistribution(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, xlsx,regionName,showImages,city=regionName)

    return model

def ApplyTraining(xlsx, regionName, model,showImages,city):

    trainData, testData, monthTrainData, monthTestData, split = splitSpeiData(xlsx)

    trainDataForPrediction, trainDataTrueValues = cria_IN_OUT(trainData, totalPoints)
    testDataForPrediction, testDataTrueValues = cria_IN_OUT(testData, totalPoints)

    trainMonthsForPrediction, trainMonthForPredictedValues = cria_IN_OUT(monthTrainData, totalPoints)
    testMonthsForPrediction, testMonthForPredictedValues = cria_IN_OUT(monthTestData, totalPoints)

    trainPredictValues = model.predict(trainDataForPrediction)
    testPredictValues = model.predict(testDataForPrediction)

    trainErrors = getError(trainDataTrueValues, trainPredictValues)
    testErrors = getError(testDataTrueValues, testPredictValues)
    
    if city not in metricsCompendium:
        metricsCompendium[city] = {}
    if regionName not in metricsCompendium[city]:
        metricsCompendium[city][regionName] = {"trainErrors": [], "testErrors": []}
        
    metricsCompendium[city][regionName]["trainErrors"].append(trainErrors)
    metricsCompendium[city][regionName]["testErrors"].append(testErrors)
    
    print("--------------Result for " +  regionName + "---------------")
    print("---------------------Train-----------------------")
    print(trainErrors)

    print("---------------------Test------------------------")
    print(testErrors)

    showSpeiData(xlsx, testData, split, regionName,showImages,city)
    showPredictionResults(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, trainMonthForPredictedValues, testMonthForPredictedValues, xlsx,regionName,showImages,city)
    showPredictionsDistribution(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, xlsx,regionName,showImages,city)
    plt.close()

def PrintmetricsCompendium():
    import pandas as pd
    # Lista de registros da planilha
    registros = []

    # Percorrendo os dados e adicionando os registros à lista
    for regiao, cidades in metricsCompendium.items():
        for cidade, metrics in cidades.items():
            # Adicionando registro de treinamento
            registro_treinamento = [f'{regiao}/{cidade}', 
                                    metrics['trainErrors'][0]['MAE'], 
                                    metrics['testErrors'][0]['MAE'], 
                                    metrics['trainErrors'][0]['RMSE'], 
                                    metrics['testErrors'][0]['RMSE'],
                                    metrics['trainErrors'][0]['MSE'], 
                                    metrics['testErrors'][0]['MSE'], 
                                    metrics['trainErrors'][0]['R^2'], 
                                    metrics['testErrors'][0]['R^2']]
            registros.append(registro_treinamento)

    # Criando DataFrame com os registros
    df = pd.DataFrame(registros, columns=['Municipio Treinado/Municipio Previsto', 'MAE Treinamento', 'MAE Validação', 'RMSE Treinamento', 'RMSE Validação', 'MSE Treinamento', 'MSE Validação', 'R^2 Treinamento', 'R^2 Validação'])

    # Escrevendo DataFrame em um arquivo Excel
    df.to_excel('metricas_modelo.xlsx', index=False)

