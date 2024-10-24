import matplotlib.pyplot as plt
import numpy as np
import statistics

from NeuralNetwork.DataProcess import readXlsx

def saveFig(plot, filename, city_cluster_name=None, city_for_training=None, city_for_predicting=None):
    if city_for_predicting:
        FILEPATH = f'./Images/cluster {city_cluster_name}/model {city_for_training}/city {city_for_predicting}/'
        plt.savefig(FILEPATH + filename + f' - Model {city_for_training} applied to {city_for_predicting}.png')
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

def DrawModelLineGraph(history, city_cluster_name, city_for_training, showImages):
    
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    
    axs[0, 0].plot(history.history['mae'] , 'tab:blue')
    axs[0, 0].set_title('MAE')
    axs[0, 0].legend(['loss'])
    
    axs[0, 1].plot(history.history['rmse'], 'tab:orange')
    axs[0, 1].set_title('RMSE')
    axs[0, 1].legend(['loss'])
    
    axs[1, 0].plot(history.history['mse'] , 'tab:green')
    axs[1, 0].set_title('MSE')
    axs[1, 0].legend(['loss'])
    
    axs[1, 1].plot(history.history['r2']  , 'tab:red')
    axs[1, 1].set_title('R²')
    axs[1, 1].legend(['explanation power'])
    
    for ax in axs[1]: # axs[1] = 2nd row
        ax.set(xlabel='Epochs (training)')
    
    plt.suptitle(f'Model {city_for_training}')

    if(showImages):
        plt.show()
    
    saveFig(plt, 'Line Graph.', city_cluster_name, city_for_training)
    plt.close()

def define_box_properties(plot_name, color_code, label):
	for k, v in plot_name.items():
		plt.setp(plot_name.get(k), color=color_code)
		
	# use plot function to draw a small line to name the legend.
	plt.plot([], c=color_code, label=label)
	plt.legend()

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
        training_plot   = plt.boxplot(metrics_dict[metric_name]['Treinamento'].values(), positions=np.array(np.arange(len(metrics_dict[metric_name]['Treinamento' ].values())))*2.0-0.35)
        validation_plot = plt.boxplot(metrics_dict[metric_name]['Validação'  ].values(), positions=np.array(np.arange(len(metrics_dict[metric_name]['Validação'   ].values())))*2.0+0.35)
    
        # setting colors for each groups
        define_box_properties(training_plot  , '#D7191C', 'Training'  )
        define_box_properties(validation_plot, '#2C7BB6', 'Validation')
    
        # set the x label values
        plt.xticks(np.arange(0, len(metrics_dict[metric_name]['Validação'].keys()) * 2, 2), metrics_dict[metric_name]['Validação'].keys(), rotation=45)
        
        plt.title(f'Comparison of performance of different models ({metric_name})')
        plt.xlabel('Machine Learning models')
        plt.ylabel(f'{metric_name} values')
        plt.grid(axis='y', linestyle=':', color='gray', linewidth=0.7)
        
        if(showImages):
            plt.show()
        
        saveFig(plt, f'Box Plots. {metric_name}.')
        plt.close()

def DrawMetricsBarPlots(metrics_df, showImages):
    metrics_df = metrics_df.drop('Agrupamento', axis='columns') # Clustering isn't much important for OneToMany, as it is redundant with 'Municipio Treinado'. It is, however, very important for ManyToMany.
    
    # Creation of the empty dictionary:
    list_of_metrics_names = ['MAE', 'RMSE', 'MSE', 'R^2']
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
        Y_axis = np.arange(len(list_of_models_names)) 
        
        # 0.4: width of the bars; 0.2: distance between the groups
        plt.barh(Y_axis - 0.2, metrics_averages_dict[metric_name]['Treinamento'].values(), 0.4, label = 'Training')
        plt.barh(Y_axis + 0.2, metrics_averages_dict[metric_name]['Validação']  .values()  , 0.4, label = 'Validation')
        
        plt.yticks(Y_axis, list_of_models_names, rotation=45)
        plt.ylabel("Machine Learning models")
        plt.xlabel(f'Average {metric_name}' if metric_name != 'R^2' else 'Average R²')
        plt.title ("Comparison of performance of different models")
        plt.legend()
        
        if(showImages):
            plt.show()
        
        saveFig(plt, f'Bar Plots. {metric_name}.')
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
        for model_name in list_of_models_names:
            x = [ metrics_dict[metric_name]['Treinamento'][model_name], metrics_dict[metric_name]['Validação'][model_name] ]
            plt.hist(x, density=True, histtype='bar', color=['red', 'green'], label=['Treinamento', 'Validação'])
            plt.legend()
            plt.title(f'Histogram of {metric_name} of model {model_name}')
            
            if(showImages):
                plt.show()
            
            saveFig(plt, f'Histograms. {metric_name}. {metric_type}.', model_name, model_name)
            plt.close()

def ShowResidualPlots(true_values, predicted_values, dataset_type, city_cluster_name, city_for_training, city_for_predicting, showImages):
    residuals = true_values - predicted_values
    
    plt.scatter(predicted_values, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {dataset_type} Data. Model {city_for_training} applied to {city_for_predicting}.')
    if(showImages):
        plt.show()
    
    saveFig(plt, f'Residual Plots {dataset_type}', city_cluster_name, city_for_training, city_for_predicting)
    plt.close()