import matplotlib.pyplot as plt
import numpy as np

from NeuralNetwork.DataProcess import getMonthValues, getSpeiValues

def saveFig(plot, filename, city_cluster_name, city_for_training, city_for_predicting=None):
    if city_for_predicting:
        FILEPATH = f'./Images/cluster {city_cluster_name}/model {city_for_training}/city {city_for_predicting}/'
        plt.savefig(FILEPATH + filename + f' - Model {city_for_training} applied to {city_for_predicting}')
    else:
        FILEPATH = f'./Images/cluster {city_cluster_name}/model {city_for_training}/'
        plt.savefig(FILEPATH + filename + f' - Model {city_for_training}')

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
    
    saveFig(plt, 'SPEI Data', city_cluster_name, city_for_training, city_for_predicting)
    plt.close()
    
def showSpeiTest(xlsx, test_data, split, city_cluster_name, city_for_training, city_for_predicting, showImages):
    
    monthValues = getMonthValues(xlsx)
    speiValues, speiNormalizedValues =  getSpeiValues(xlsx)

    y1positive=np.array(speiValues)>=0
    y1negative = np.array(speiValues)<=0

    plt.figure()
    plt.fill_between(monthValues, speiValues,y2=0,where=y1positive,
    color='green',alpha=0.5,interpolate=False, label='índices SPEI positivos')
    plt.fill_between(monthValues, speiValues,y2=0,where=y1negative,
    color='red',alpha=0.5,interpolate=False, label='índices SPEI negativos')
    plt.xlabel('Ano')
    plt.ylabel('SPEI')
    plt.title(f'{city_for_predicting}: SPEI Data')
    plt.legend()
    if(showImages):
        plt.show()
    
    saveFig(plt, 'SPEI Data (test)', city_cluster_name, city_for_training, city_for_predicting)
    plt.close()
    
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
    plt.title(f'{city_for_predicting}: valores verdadeiros e previstos para o final das séries')
    if(showImages):
        plt.show()
    
    saveFig(plt, 'Previsao', city_cluster_name, city_for_training, city_for_predicting)
    plt.close()
    
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
    plt.title(f'{city_for_predicting}: SPEI (distribuição)')
    
    if(showImages):
        plt.show()
        
    saveFig(plt, 'distribuiçãoDoSPEI', city_cluster_name, city_for_training, city_for_predicting)
    plt.close()

def showNeuralNetworkModelMetrics(history, city_cluster_name, city_for_training, showImages):
    # MAE plot:
    plt.figure()
    plt.plot(history.history['mae'],'k')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend(['loss'])
    plt.title(f'{city_for_training}: MAE')
    
    if(showImages):
        plt.show()
    
    saveFig(plt, 'MAE', city_cluster_name, city_for_training)
    plt.close()
    
    # RMSE plot:
    plt.figure()
    plt.plot(history.history['root_mean_squared_error'],'k')
    plt.xlabel('Epochs')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.legend(['loss'])
    plt.title(f'{city_for_training}: RMSE')
    
    if(showImages):
        plt.show()
    
    saveFig(plt, 'RMSE', city_cluster_name, city_for_training)
    plt.close()
    
    # MSE plot:
    plt.figure()
    plt.plot(history.history['mse'],'k')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend(['loss'])
    plt.title(f'{city_for_training}: MSE')
    
    if(showImages):
        plt.show()
    
    saveFig(plt, 'MSE', city_cluster_name, city_for_training)
    plt.close()

def DrawMetricsBoxPlots(metrics_df, showImages):
    metrics_df = metrics_df.drop('Agrupamento', axis='columns') # Clustering isn't much important for OneToMany, as it is redundant with 'Municipio Treinado'. It is, however, very important for ManyToMany.
    list_of_cities_for_training = metrics_df['Municipio Treinado'].unique()

    import matplotlib.pyplot as plt
    MAE_dict  = {}
    RMSE_dict = {}
    MSE_dict  = {}

    for model_name in list_of_cities_for_training:
        MAE_dict [model_name] = {}
        RMSE_dict[model_name] = {}
        MSE_dict [model_name] = {}
        
        metrics_current_model_df = metrics_df[ metrics_df['Municipio Treinado'] == model_name ]
        
        MAE_dict[model_name]['MAE Treinamento'] = metrics_current_model_df['MAE Treinamento'].to_list()
        MAE_dict[model_name]['MAE Validação']   = metrics_current_model_df['MAE Validação']  .to_list()

        RMSE_dict[model_name]['RMSE Treinamento'] = metrics_current_model_df['RMSE Treinamento'].to_list()
        RMSE_dict[model_name]['RMSE Validação']   = metrics_current_model_df['RMSE Validação']  .to_list()
        
        MSE_dict [model_name]['MSE Treinamento'] = metrics_current_model_df['MSE Treinamento'].to_list()
        MSE_dict [model_name]['MSE Validação']   = metrics_current_model_df['MSE Validação']  .to_list()
        
        # MAE plot:
        plt.boxplot([MAE_dict[model_name]['MAE Treinamento'], MAE_dict[model_name]['MAE Validação']], tick_labels=['MAE training', 'MAE testing'])
        plt.title('Box Plots of Metrics')
        plt.ylabel('Metric Value')
        plt.title(f'MAE of model {model_name}')
        plt.grid(axis='y', linestyle=':', color='gray', linewidth=0.7)
        
        if(showImages):
            plt.show()
        
        saveFig(plt, 'MAE Boxplots', model_name, model_name)
        plt.close()
        
        # RMSE plot:
        plt.boxplot([RMSE_dict[model_name]['RMSE Treinamento'], RMSE_dict[model_name]['RMSE Validação']], tick_labels=['RMSE training', 'RMSE testing'])
        plt.title('Box Plots of Metrics')
        plt.ylabel('Metric Value')
        plt.title(f'RMSE of model {model_name}')
        plt.grid(axis='y', linestyle=':', color='gray', linewidth=0.7)
        
        if(showImages):
            plt.show()
        
        saveFig(plt, 'RMSE Boxplots', model_name, model_name)
        plt.close()
        
        # MSE plot:
        plt.boxplot([MSE_dict[model_name]['MSE Treinamento'], MSE_dict[model_name]['MSE Validação']], tick_labels=['MSE training', 'MSE testing'])
        plt.title('Box Plots of Metrics')
        plt.ylabel('Metric Value')
        plt.title(f'MSE of model {model_name}')
        plt.grid(axis='y', linestyle=':', color='gray', linewidth=0.7)
        
        if(showImages):
            plt.show()
        
        saveFig(plt, 'MSE Boxplots', model_name, model_name)
        plt.close()