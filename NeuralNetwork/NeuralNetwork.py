import tensorflow as tf
import json
import matplotlib.pyplot as plt

from NeuralNetwork.DataProcess import splitSpeiData, cria_IN_OUT
from NeuralNetwork.VisualRepresentation import showPredictionResults, showPredictionsDistribution, showSpeiData, showSpeiTest, DrawModelsLineGraph, ShowResidualPlots
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
    model.compile(loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'mse'], optimizer='adam')
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
    
    DrawModelsLineGraph(history, city_cluster_name, city_for_training, showImages)
    
    return model

def UseNeuralNetwork(xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages, model=None, training=True):
        #[0] = lista de dados do SPEI referentes à parcela de treinamento (80%)
        #[1] = lista de dados do SPEI referentes à parcela de teste (20%)
        #[2] = lista de datas referentes à parcela de treinamento (80%)
        #[3] = lista de datas referentes à parcela de teste (20%)
        #[4] = valor inteiro da posição que o dataset foi splitado
    trainData, testData, monthTrainData, monthTestData, split = splitSpeiData(xlsx)

        # Dataset que contém a parcela de dados que será utilizadda para...
        #[0] = ... alimentar a predição da rede(treinamento)
        #[1] = ... validar se as predições da rede estão corretas(treinamento)
    trainDataForPrediction, trainDataTrueValues = cria_IN_OUT(trainData, totalPoints)

        # Dataset que contém a parcela de dados que será utilizadda para...
        #[0] = ... alimentar a predição da rede(teste)
        #[1] = ... validar se as predições da rede estão corretas(teste)
    testDataForPrediction, testDataTrueValues = cria_IN_OUT(testData, totalPoints)

        # Dataset que contém a parcela dos mses nos quais...
        #[0] = ... os SPEIs foram utilizados para alimentar a predição da rede(treinamento)
        #[1] = ... os SPEIs foram preditos(treinamento)
    trainMonthsForPrediction, trainMonthForPredictedValues = cria_IN_OUT(monthTrainData, totalPoints)

        # Dataset que contém a parcela dos mses nos quais...
        #[0] = ... os SPEIs foram utilizados para alimentar a predição da rede(teste)
        #[1] = ... os SPEIs foram preditos(teste)
    testMonthsForPrediction, testMonthForPredictedValues = cria_IN_OUT(monthTestData, totalPoints)

    if training:
        model = trainNeuralNetwork(trainDataForPrediction, trainDataTrueValues, showImages, city_cluster_name, city_for_training, city_for_predicting)

        #faz previsões e calcula os erros
    trainPredictValues = model.predict(trainDataForPrediction, verbose = 0)
    testPredictValues  = model.predict(testDataForPrediction , verbose = 0)

    trainErrors = getError(trainDataTrueValues, trainPredictValues)
    testErrors  = getError(testDataTrueValues , testPredictValues)
    
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

    ShowResidualPlots(trainDataTrueValues, trainPredictValues, 'Training', city_cluster_name, city_for_training, city_for_predicting, showImages)
    ShowResidualPlots(testDataTrueValues , testPredictValues , 'Testing' , city_cluster_name, city_for_training, city_for_predicting, showImages)
    
    showSpeiData(xlsx, testData, split, city_cluster_name, city_for_training, city_for_predicting, showImages)
    if training:
        showSpeiTest(xlsx, testData, split, city_cluster_name, city_for_training, city_for_predicting, showImages)
        
    showPredictionResults(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, trainMonthForPredictedValues, testMonthForPredictedValues, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages)
    showPredictionsDistribution(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages)

    return model, metricsCompendium

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

    return df