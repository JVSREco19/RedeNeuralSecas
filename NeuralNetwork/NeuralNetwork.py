import tensorflow as tf
import pandas     as pd
import json

from NeuralNetwork.DataProcess import splitSpeiData, cria_IN_OUT
from NeuralNetwork.VisualRepresentation import showPredictionResults, showPredictionsDistribution, showSpeiData, showSpeiTest, drawModelLineGraph, showResidualPlots, showR2ScatterPlots, showTaylorDiagrams
from NeuralNetwork.Metrics import getError

metricsCompendium = {}

# Abra o arquivo JSON
with open("NeuralNetwork/config.json") as arquivo:
    dados_json = json.load(arquivo)

totalPoints      = dados_json['totalPoints']
predictionPoints = dados_json['predictionPoints']
numberOfEpochs   = dados_json['numberOfEpochs']
hiddenUnits      = dados_json['hiddenUnits']

def createNeuralNetwork(hidden_units, dense_units, input_shape, activation):
    model = tf.keras.Sequential()   
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.LSTM(hidden_units,activation=activation[0]))
    model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
    model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
    model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
    model.compile(loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'mse', tf.keras.metrics.R2Score(name="r2")], optimizer='adam')
    return model

def trainNeuralNetwork(trainDataForPrediction, trainDataTrueValues, showImages, city_cluster_name, city_for_training, city_for_predicting):
    model = createNeuralNetwork( hidden_units= hiddenUnits, dense_units=predictionPoints, 
                                input_shape=(totalPoints-predictionPoints,1), activation=['relu','sigmoid'])
    print(f'\tCreated neural network model for {city_for_training}.')

    #treina a rede e mostra o gráfico do loss
    print(f'\t\tFitting neural network model for {city_for_training}...')
    history=model.fit(trainDataForPrediction, trainDataTrueValues, epochs=numberOfEpochs, batch_size=1, verbose=0)
    print(f'\t\tFitted neural network model for {city_for_training}.')
    
    drawModelLineGraph(history, city_cluster_name, city_for_training, showImages)
    
    return model

def UseNeuralNetwork(xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages, model=None, training=True):
        #[0] = lista de dados do SPEI referentes à parcela de treinamento (80%)
        #[1] = lista de dados do SPEI referentes à parcela de teste       (20%)
        #[2] = lista de datas referentes à parcela de treinamento         (80%)
        #[3] = lista de datas referentes à parcela de teste               (20%)
        #[4] = valor inteiro da posição que o dataset foi splitado
    trainData, testData, monthTrainData, monthTestData, split = splitSpeiData(xlsx)

        # Dataset que contém a parcela de dados que será utilizada para...
        #[0] = ... alimentar a predição da rede
        #[1] = ... validar se as predições da rede estão corretas
    trainDataForPrediction, trainDataTrueValues = cria_IN_OUT(trainData, totalPoints) # Treinamento
    testDataForPrediction , testDataTrueValues  = cria_IN_OUT(testData , totalPoints) # Teste

        # Dataset que contém a parcela dos meses nos quais...
        #[0] = ... os SPEIs foram utilizados para alimentar a predição da rede
        #[1] = ... os SPEIs foram preditos
    trainMonthsForPrediction, trainMonthForPredictedValues = cria_IN_OUT(monthTrainData, totalPoints) # Treinamento
    testMonthsForPrediction , testMonthForPredictedValues  = cria_IN_OUT(monthTestData , totalPoints) # Teste

    if training:
        model = trainNeuralNetwork(trainDataForPrediction, trainDataTrueValues, showImages, city_cluster_name, city_for_training, city_for_predicting)

    trainPredictValues = model.predict(trainDataForPrediction, verbose = 0)
    testPredictValues  = model.predict(testDataForPrediction , verbose = 0)

    # RMSE, MSE, MAE, R²:
    trainErrors = getError(trainDataTrueValues, trainPredictValues)
    testErrors  = getError(testDataTrueValues , testPredictValues )
    
    if city_cluster_name not in metricsCompendium:
        metricsCompendium[city_cluster_name] = {}
    if city_for_training not in metricsCompendium[city_cluster_name]:
        metricsCompendium[city_cluster_name][city_for_training] = {}
    if city_for_predicting not in metricsCompendium[city_cluster_name][city_for_training]:
        metricsCompendium[city_cluster_name][city_for_training][city_for_predicting] = {'trainErrors': [], 'testErrors': []}
    
    metricsCompendium[city_cluster_name][city_for_training][city_for_predicting]['trainErrors'] = trainErrors
    metricsCompendium[city_cluster_name][city_for_training][city_for_predicting]['testErrors' ] = testErrors

    if training == True:
        print(f'\t\t--------------Result for {city_for_training} (training)---------------')
    else:
        print(f'\t\t--------------Result for {city_for_training} applied to {city_for_predicting}---------------')
    print(f'\t\t\tTRAIN: {trainErrors}')
    print(f'\t\t\tTEST : {testErrors} ')
        
    # Plots:
    showTaylorDiagrams(trainErrors['RMSE'], testErrors['RMSE'], trainData, testData, trainDataTrueValues, trainPredictValues, testDataTrueValues, testPredictValues, city_cluster_name, city_for_training, city_for_predicting, showImages)
    showResidualPlots(trainDataTrueValues, trainPredictValues, testDataTrueValues, testPredictValues, city_cluster_name, city_for_training, city_for_predicting, showImages)
    showR2ScatterPlots(trainDataTrueValues, trainPredictValues, testDataTrueValues, testPredictValues, city_cluster_name, city_for_training, city_for_predicting, showImages)
    
    showSpeiData(xlsx, testData, split, city_cluster_name, city_for_training, city_for_predicting, showImages)
    if training:
        showSpeiTest(xlsx, testData, split, city_cluster_name, city_for_training, city_for_predicting, showImages)
        
    showPredictionResults(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, trainMonthForPredictedValues, testMonthForPredictedValues, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages)
    showPredictionsDistribution(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages)

    return model, metricsCompendium

def PrintMetricsList(metricsCompendium):   
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
                list_of_all_metrics_city_by_city.append(list_of_metrics_of_one_city)
    
    df = pd.DataFrame(list_of_all_metrics_city_by_city,
                      columns=['Agrupamento', 'Municipio Treinado', 'Municipio Previsto', 'MAE Treinamento', 'MAE Validação', 'RMSE Treinamento', 'RMSE Validação', 'MSE Treinamento', 'MSE Validação', 'R^2 Treinamento', 'R^2 Validação'])

    # Escrevendo DataFrame em um arquivo Excel
    df.to_excel('metricas_modelo.xlsx', index=False)

    return df