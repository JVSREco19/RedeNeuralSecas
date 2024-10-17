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

totalPoints      = dados_json['totalPoints']
predictionPoints = dados_json['predictionPoints']
numberOfEpochs   = dados_json['numberOfEpochs']
hiddenUnits      = dados_json['hiddenUnits']

def createNeuralNetwork(hidden_units, dense_units, input_shape, activation):
    model = tf.keras.Sequential()   
    model.add(tf.keras.layers.LSTM(hidden_units,input_shape=input_shape,activation=activation[0]))
    model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
    model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
    model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
    model.compile(loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(), 'mse'], optimizer='adam')
    #to be added: tf.keras.metrics.R2Score(name='r2_score', dtype=tf.float32)
    return model

def trainNeuralNetwork(trainDataForPrediction, trainDataTrueValues, showImages, city_cluster_name, city_for_training, city_for_predicting):

    model = createNeuralNetwork( hidden_units= hiddenUnits, dense_units=predictionPoints, 
                                input_shape=(totalPoints-predictionPoints,1), activation=['relu','sigmoid'])
    #print(model.summary())
    print(f'\tCreated neural network model for {city_for_training}.')

    #treina a rede e mostra o gráfico do loss
    print(f'\t\tFitting neural network model for {city_for_training}...')
    history=model.fit(trainDataForPrediction, trainDataTrueValues, epochs=numberOfEpochs, batch_size=1, verbose=0)
    print(f'\t\tFitted neural network model for {city_for_training}.')
    
    plt.figure()
    plt.plot(history.history['mae'],'k')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend(['loss'])
    
    if(showImages):
        plt.show()
    plt.savefig(f'./Images/cluster {city_cluster_name}/model {city_for_training}/MAE')
    plt.close()
    
    plt.figure()
    plt.plot(history.history['root_mean_squared_error'],'k')
    plt.xlabel('Epochs')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.legend(['loss'])
    
    if(showImages):
        plt.show()
    plt.savefig(f'./Images/cluster {city_cluster_name}/model {city_for_training}/RMSE')
    plt.close()
    
    plt.figure()
    plt.plot(history.history['mse'],'k')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend(['loss'])
    
    if(showImages):
        plt.show()
    plt.savefig(f'./Images/cluster {city_cluster_name}/model {city_for_training}/MSE')
    plt.close()
    
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

def FitNeuralNetwork(xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages):

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

        #[0] = Dataset que contem a parcela dos meses nos quais os SPEIs foram utilizados para alimentar a predição da rede(treinamento)
        #[1] = Dataset que contem a parcela dos meses nos quais os SPEIs foram preditos(treinamento)
    trainMonthsForPrediction, trainMonthForPredictedValues = cria_IN_OUT(monthTrainData, totalPoints)

        #[0] = Dataset que contem a parcela dos meses nos quais os SPEIs foram utilizados para alimentar a predição da rede(teste)
        #[1] = Dataset que contem a parcela dos meses nos quais os SPEIs foram preditos(teste)
    testMonthsForPrediction, testMonthForPredictedValues = cria_IN_OUT(monthTestData, totalPoints)

    model = trainNeuralNetwork(trainDataForPrediction, trainDataTrueValues, showImages, city_cluster_name, city_for_training, city_for_predicting)

        #faz previsões e calcula os erros
    trainPredictValues = model.predict(trainDataForPrediction, verbose = 0)
    testPredictValues  = model.predict(testDataForPrediction , verbose = 0)

    trainErrors = getError(trainDataTrueValues, trainPredictValues)
    testErrors = getError(testDataTrueValues, testPredictValues)
    
    if city_cluster_name not in metricsCompendium:
        metricsCompendium[city_cluster_name] = {}
    if city_for_training not in metricsCompendium[city_cluster_name]:
        metricsCompendium[city_cluster_name][city_for_training] = {}
    if city_for_predicting not in metricsCompendium[city_cluster_name][city_for_training]:
        metricsCompendium[city_cluster_name][city_for_training][city_for_predicting] = {"trainErrors": [], "testErrors": []}
    
    metricsCompendium[city_cluster_name][city_for_training][city_for_predicting]["trainErrors"] = trainErrors
    metricsCompendium[city_cluster_name][city_for_training][city_for_predicting]["testErrors"]  = testErrors

    print("\t\t--------------Result for " + city_for_training +"---------------")
    print(f'\t\t\tTRAIN: {trainErrors}')    
    print(f'\t\t\tTEST : {testErrors}')

    showSpeiData(xlsx, testData, split, city_cluster_name, city_for_training, city_for_predicting, showImages)
    showSpeiTest(xlsx, testData, split, city_cluster_name, city_for_training, city_for_predicting, showImages)
    showPredictionResults(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, trainMonthForPredictedValues, testMonthForPredictedValues, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages)
    showPredictionsDistribution(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages)

    return model, metricsCompendium

def ApplyTraining(xlsx, city_cluster_name, city_for_training, city_for_predicting, model, showImages, city):

    trainData, testData, monthTrainData, monthTestData, split = splitSpeiData(xlsx)

    trainDataForPrediction, trainDataTrueValues = cria_IN_OUT(trainData, totalPoints)
    testDataForPrediction, testDataTrueValues = cria_IN_OUT(testData, totalPoints)

    trainMonthsForPrediction, trainMonthForPredictedValues = cria_IN_OUT(monthTrainData, totalPoints)
    testMonthsForPrediction, testMonthForPredictedValues = cria_IN_OUT(monthTestData, totalPoints)

    trainPredictValues = model.predict(trainDataForPrediction, verbose = 0)
    testPredictValues  = model.predict(testDataForPrediction , verbose = 0)

    trainErrors = getError(trainDataTrueValues, trainPredictValues)
    testErrors = getError(testDataTrueValues, testPredictValues)
    
    if city_cluster_name not in metricsCompendium:
        metricsCompendium[city_cluster_name] = {}
    if city_for_training not in metricsCompendium[city_cluster_name]:
        metricsCompendium[city_cluster_name][city_for_training] = {}
    if city_for_predicting not in metricsCompendium[city_cluster_name][city_for_training]:
        metricsCompendium[city_cluster_name][city_for_training][city_for_predicting] = {"trainErrors": [], "testErrors": []}
    
    metricsCompendium[city_cluster_name][city_for_training][city_for_predicting]["trainErrors"] = trainErrors
    metricsCompendium[city_cluster_name][city_for_training][city_for_predicting]["testErrors"]  = testErrors
    
    print("\t--------------Result for " +  city_for_training + "'s " + city_for_predicting + "---------------")
    print(f'\t\tTRAIN: {trainErrors}')    
    print(f'\t\tTEST : {testErrors}')

    showSpeiData(xlsx, testData, split, city_cluster_name, city_for_training, city_for_predicting, showImages)
    showPredictionResults(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, trainMonthForPredictedValues, testMonthForPredictedValues, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages)
    showPredictionsDistribution(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages)
    plt.close()
    
    return metricsCompendium

def PrintMetricsList(metricsCompendium):
    import pandas as pd
    
    list_of_all_metrics_city_by_city = []
    
    for city_cluster_name, dict_of_central_cities in metricsCompendium.items():
        for central_city_name, dict_of_bordering_cities in dict_of_central_cities.items():
            for bordering_city_name, dict_of_measurement_types in dict_of_bordering_cities.items():
                list_of_metrics_of_one_city = [f'{city_cluster_name}',
                                               f'{central_city_name}',
                                               f'{bordering_city_name}',
                                               dict_of_measurement_types['testErrors' ]['MAE'],
                                               dict_of_measurement_types['trainErrors']['MAE'],
                                               
                                               dict_of_measurement_types['testErrors' ]['RMSE'],
                                               dict_of_measurement_types['trainErrors']['RMSE'],
                                               
                                               dict_of_measurement_types['testErrors' ]['MSE'],
                                               dict_of_measurement_types['trainErrors']['MSE'],
                                               
                                               dict_of_measurement_types['testErrors' ]['R^2'],
                                               dict_of_measurement_types['trainErrors']['R^2']
                                               ]
         #       print(list_of_metrics_of_one_city)
                list_of_all_metrics_city_by_city.append(list_of_metrics_of_one_city)
    
    #print(list_of_metrics_of_one_city)
    
    df = pd.DataFrame(list_of_all_metrics_city_by_city,
                      columns=['Agrupamento', 'Municipio Treinado', 'Municipio Previsto', 'MAE Treinamento', 'MAE Validação', 'RMSE Treinamento', 'RMSE Validação', 'MSE Treinamento', 'MSE Validação', 'R^2 Treinamento', 'R^2 Validação'])

    # Escrevendo DataFrame em um arquivo Excel
    df.to_excel('metricas_modelo.xlsx', index=False)
