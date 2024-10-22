import matplotlib.pyplot as plt
import numpy as np
import statistics
import pprint

from NeuralNetwork.DataProcess import readXlsx

def saveFig(plot, filename, city_cluster_name=None, city_for_training=None, city_for_predicting=None):
    if city_for_predicting:
        FILEPATH = f'./Images/cluster {city_cluster_name}/model {city_for_training}/city {city_for_predicting}/'
        plt.savefig(FILEPATH + filename + f' - Model {city_for_training} applied to {city_for_predicting}')
    elif city_for_training:
        FILEPATH = f'./Images/cluster {city_cluster_name}/model {city_for_training}/'
        plt.savefig(FILEPATH + filename + f' - Model {city_for_training}.png')
    else:
        FILEPATH = './Images/'
        plt.savefig(FILEPATH + filename, bbox_inches="tight")

def showSpeiData(xlsx, test_data, split, city_cluster_name, city_for_training, city_for_predicting, showImages):
    
    speiValues, speiNormalizedValues, monthValues =  readXlsx(xlsx)
    
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
    
    speiValues, speiNormalizedValues, monthValues =  readXlsx(xlsx)

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

    SpeiValues, SpeiNormalizedValues, monthValues = readXlsx(xlsx)

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

    SpeiValues, SpeiNormalizedValues, monthValues = readXlsx(xlsx)

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

def DrawModelsLineGraph(history, city_cluster_name, city_for_training, showImages):
    metrics_dict = {'mae' : 'Mean Absolute Error',
                    'rmse': 'Root Mean Squared Error',
                    'mse' : 'Mean Squared Error'
                    }
    
    for metric_shortname, metric_longname in metrics_dict.items():
        plt.figure()
        plt.plot(history.history[metric_shortname],'k')
        plt.xlabel('Epochs')
        plt.ylabel(f'{metric_longname} ({metric_shortname.upper()})')
        plt.legend(['loss'])
        plt.title(f'{city_for_training}: {metric_shortname.upper()}')
        
        if(showImages):
            plt.show()
        
        saveFig(plt, f'Line Graph. {metric_shortname.upper()}', city_cluster_name, city_for_training)
        plt.close()
    
def DrawMetricsBoxPlots(metrics_df, showImages):
    metrics_df = metrics_df.drop('Agrupamento', axis='columns') # Clustering isn't much important for OneToMany, as it is redundant with 'Municipio Treinado'. It is, however, very important for ManyToMany.
    
    # Creation of the empty dictionary:
    list_of_metrics_names = ['MAE', 'RMSE', 'MSE']
    list_of_metrics_types = ['Treinamento', 'Validação']
    list_of_models_names  = metrics_df['Municipio Treinado'].unique()
    
    metrics_dict = dict.fromkeys(list_of_metrics_names)
    for metric_name in metrics_dict.keys():
        metrics_dict[metric_name] = dict.fromkeys(list_of_metrics_types)
        
        for metric_type in metrics_dict[metric_name].keys():
            metrics_dict[metric_name][metric_type] = dict.fromkeys(list_of_models_names)
    
    # Filling the dictionary:
    for metric_name in list_of_metrics_names:
        for metric_type in list_of_metrics_types:
            for model_name in list_of_models_names:
                metrics_dict[metric_name][metric_type][model_name] = metrics_df[ metrics_df['Municipio Treinado'] == model_name ][f'{metric_name} {metric_type}'].to_list()
    
    # Plotting the graphs:
    for metric_name in list_of_metrics_names:
        for metric_type in list_of_metrics_types:
            plt.boxplot(metrics_dict[metric_name][metric_type].values(), tick_labels=metrics_dict[metric_name][metric_type].keys())
            plt.xlabel('Machine Learning models')
            plt.ylabel(f'{metric_name} {metric_type} values')
            plt.title(f'Comparison of performance of different models ({metric_name})')
            plt.grid(axis='y', linestyle=':', color='gray', linewidth=0.7)
            plt.xticks(rotation=45)
            if(showImages):
                plt.show()
            
            saveFig(plt, f'Box Plots. {metric_name}. {metric_type}.')
            plt.close()

def DrawMetricsBarPlots(metrics_df, showImages):
    metrics_df = metrics_df.drop('Agrupamento', axis='columns') # Clustering isn't much important for OneToMany, as it is redundant with 'Municipio Treinado'. It is, however, very important for ManyToMany.
    
    # Creation of the empty dictionary:
    list_of_metrics_names = ['MAE', 'RMSE', 'MSE'] # To-do: implement also R²
    list_of_metrics_types = ['Treinamento', 'Validação']
    list_of_models_names  = metrics_df['Municipio Treinado'].unique()
    
    metrics_averages_dict = dict.fromkeys(list_of_metrics_names)
    for metric_name in metrics_averages_dict.keys():
        metrics_averages_dict[metric_name] = dict.fromkeys(list_of_metrics_types)
        
        for metric_type in metrics_averages_dict[metric_name].keys():
            metrics_averages_dict[metric_name][metric_type] = dict.fromkeys(list_of_models_names)
    
    # Filling the dictionary:
    for metric_name in list_of_metrics_names:
        for metric_type in list_of_metrics_types:
            for model_name in list_of_models_names:
                average = statistics.mean( metrics_df[ metrics_df['Municipio Treinado'] == model_name ][f'{metric_name} {metric_type}'].to_list() )
                metrics_averages_dict[metric_name][metric_type][model_name] = average
    
    # Plotting the graphs:
    for metric_name in list_of_metrics_names:
        for metric_type in list_of_metrics_types:        
            plt.barh(metrics_averages_dict[metric_name][metric_type].keys(), metrics_averages_dict[metric_name][metric_type].values(), color ='maroon')
            plt.xlabel(f'Average {metric_name} {metric_type}')
            plt.ylabel("Machine Learning models")
            plt.title(f'Comparison of performance of different models ({metric_name})')
            if(showImages):
                plt.show()
            
            saveFig(plt, f'Bar Plots. {metric_name}. {metric_type}.')
            plt.close()

def DrawMetricsHistograms(metrics_df, showImages):
    metrics_df = metrics_df.drop('Agrupamento', axis='columns') # Clustering isn't much important for OneToMany, as it is redundant with 'Municipio Treinado'. It is, however, very important for ManyToMany.
    
    # Creation of the empty dictionary:
    list_of_metrics_names = ['MAE', 'RMSE']
    list_of_metrics_types = ['Treinamento', 'Validação']
    list_of_models_names  = metrics_df['Municipio Treinado'].unique()
    
    metrics_dict = dict.fromkeys(list_of_metrics_names)
    for metric_name in metrics_dict.keys():
        metrics_dict[metric_name] = dict.fromkeys(list_of_metrics_types)
        
        for metric_type in metrics_dict[metric_name].keys():
            metrics_dict[metric_name][metric_type] = dict.fromkeys(list_of_models_names)
    
    # Filling the dictionary:
    for metric_name in list_of_metrics_names:
        for metric_type in list_of_metrics_types:
            for model_name in list_of_models_names:
                metrics_dict[metric_name][metric_type][model_name] = metrics_df[ metrics_df['Municipio Treinado'] == model_name ][f'{metric_name} {metric_type}'].to_list()
    
    # Plotting the graphs:
    for metric_name in list_of_metrics_names:
        for metric_type in list_of_metrics_types:
            for model_name in list_of_models_names:
                plt.hist(metrics_dict[metric_name][metric_type][model_name])
                plt.title(f'Historgram of {metric_name} ({metric_type}) of model {model_name}')
                plt.xlabel(f'{metric_name} {metric_type}')
                plt.ylabel('Frequency')
                if(showImages):
                    plt.show()
                
                saveFig(plt, f'Histograms. {metric_name}. {metric_type}.', model_name, model_name)
                plt.close()