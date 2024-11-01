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

def useNeuralNetwork(xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages, model=None, training=True):
    SPEI_dict, months_dict, split = splitSpeiData(xlsx) #(SPEI/months)_dict.keys() = ['Train', 'Test']

    #         IN            ,           OUT          : 
    dataForPrediction_dict  , dataTrueValues_dict    = cria_IN_OUT(SPEI_dict  , totalPoints)
    monthsForPrediction_dict, monthForPredicted_dict = cria_IN_OUT(months_dict, totalPoints)

    if training:
        model = trainNeuralNetwork(dataForPrediction_dict['Train'], dataTrueValues_dict['Train'], showImages, city_cluster_name, city_for_training, city_for_predicting)

    predictValues_dict = {'Train': model.predict(dataForPrediction_dict['Train'], verbose = 0),
                          'Test' : model.predict(dataForPrediction_dict['Test' ], verbose = 0)
                         }

    # RMSE, MSE, MAE, R²:
    errors_dict = {'Train': getError(dataTrueValues_dict['Train'], predictValues_dict['Train']),
                   'Test' : getError(dataTrueValues_dict['Test' ], predictValues_dict['Test' ])
                  }
    
    writeErrorsOnMetricsCompendium(city_cluster_name, city_for_training, city_for_predicting, errors_dict)
    
    printErrors(errors_dict, training, city_for_training, city_for_predicting)
    
    plotModelPlots(showImages, city_cluster_name, city_for_training, city_for_predicting, errors_dict, SPEI_dict, dataTrueValues_dict, predictValues_dict, xlsx, split, monthForPredicted_dict, training)
    
    return model, metricsCompendium

def printErrors(errors_dict, training, city_for_training, city_for_predicting):
    if training == True:
        print(f'\t\t--------------Result for {city_for_training} (training)---------------')
    else:
        print(f'\t\t--------------Result for {city_for_training} applied to {city_for_predicting}---------------')
    print(f"\t\t\tTRAIN: {errors_dict['Train']}")
    print(f"\t\t\tTEST : {errors_dict['Test'] }")

def plotModelPlots(showImages, city_cluster_name, city_for_training, city_for_predicting, errors_dict, SPEI_dict, dataTrueValues_dict, predictValues_dict, xlsx, split, monthForPredicted_dict, training):
    showTaylorDiagrams(errors_dict, SPEI_dict, dataTrueValues_dict, predictValues_dict, city_cluster_name, city_for_training, city_for_predicting, showImages)
    showResidualPlots (dataTrueValues_dict, predictValues_dict, city_cluster_name, city_for_training, city_for_predicting, showImages)
    showR2ScatterPlots(dataTrueValues_dict, predictValues_dict, city_cluster_name, city_for_training, city_for_predicting, showImages)
    
    showSpeiData(xlsx, SPEI_dict['Test'], split, city_cluster_name, city_for_training, city_for_predicting, showImages)
    if training:
        showSpeiTest(xlsx, SPEI_dict['Test'], split, city_cluster_name, city_for_training, city_for_predicting, showImages)
        
    showPredictionResults      (dataTrueValues_dict, predictValues_dict, monthForPredicted_dict, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages)
    showPredictionsDistribution(dataTrueValues_dict, predictValues_dict, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages)

def writeErrorsOnMetricsCompendium(city_cluster_name, city_for_training, city_for_predicting, errors_dict):
    if city_cluster_name not in metricsCompendium:
        metricsCompendium[city_cluster_name] = {}
    if city_for_training not in metricsCompendium[city_cluster_name]:
        metricsCompendium[city_cluster_name][city_for_training] = {}
    if city_for_predicting not in metricsCompendium[city_cluster_name][city_for_training]:
        metricsCompendium[city_cluster_name][city_for_training][city_for_predicting] = {'trainErrors': [], 'testErrors': []}

    metricsCompendium[city_cluster_name][city_for_training][city_for_predicting]['trainErrors'] = errors_dict['Train']
    metricsCompendium[city_cluster_name][city_for_training][city_for_predicting]['testErrors' ] = errors_dict['Test']

def printMetricsList(metricsCompendium):   
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