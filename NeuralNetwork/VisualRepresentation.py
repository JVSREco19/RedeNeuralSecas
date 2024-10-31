import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import statistics
from scipy.stats import norm
import skill_metrics as sm

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

    y1positive = np.array(speiValues)>=0
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
    
def showPredictionResults(dataTrueValues_dict, predictValues_dict, monthForPredicted_dict, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages):
    trueValues  = np.append(dataTrueValues_dict['Train'], dataTrueValues_dict['Test'])
    predictions = np.append( predictValues_dict['Train'],  predictValues_dict['Test'])

    reshapedMonth = np.append(monthForPredicted_dict['Train'], monthForPredicted_dict['Test'])

    SpeiValues, SpeiNormalizedValues, monthValues = readXlsx(xlsx)

    speiMaxValue = np.max(SpeiValues)
    speiMinValue = np.min(SpeiValues)

    trueValues_denormalized  = (trueValues  * (speiMaxValue - speiMinValue) + speiMinValue)
    predictions_denormalized = (predictions * (speiMaxValue - speiMinValue) + speiMinValue)

    plt.figure()
    plt.plot(reshapedMonth,trueValues_denormalized)
    plt.plot(reshapedMonth,predictions_denormalized)
    plt.axvline(monthForPredicted_dict['Train'][-1][-1], color='r')
    plt.legend(['Verdadeiros', 'Previstos'])
    plt.xlabel('Data')
    plt.ylabel('SPEI')
    plt.title(f'{city_for_predicting}: valores verdadeiros e previstos para o final das séries')
    if(showImages):
        plt.show()
    
    saveFig(plt, 'Previsao', city_cluster_name, city_for_training, city_for_predicting)
    plt.close()
    
def showPredictionsDistribution(dataTrueValues_dict, predictValues_dict, xlsx, city_cluster_name, city_for_training, city_for_predicting, showImages):
    trueValues  = np.append(dataTrueValues_dict['Train'], dataTrueValues_dict['Test'])
    predictions = np.append( predictValues_dict['Train'],  predictValues_dict['Test'])

    SpeiValues, SpeiNormalizedValues, monthValues = readXlsx(xlsx)

    speiMaxValue = np.max(SpeiValues)
    speiMinValue = np.min(SpeiValues)

    trueValues_denormalized  = (trueValues  * (speiMaxValue - speiMinValue) + speiMinValue)
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

def drawModelLineGraph(history, city_cluster_name, city_for_training, showImages):
    
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

def drawMetricsBoxPlots(metrics_df, showImages):   
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

def drawMetricsBarPlots(metrics_df, showImages):
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

def define_normal_distribution(axis, x_values):
    mu, std = norm.fit(x_values)
    xmin, xmax = axis.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    
    return x, p

def drawMetricsHistograms(metrics_df, showImages):
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
    for model_name in list_of_models_names:
        x_MAE  = [ metrics_dict['MAE' ]['Treinamento'][model_name], metrics_dict['MAE' ]['Validação'][model_name] ]
        x_RMSE = [ metrics_dict['RMSE']['Treinamento'][model_name], metrics_dict['RMSE']['Validação'][model_name] ]
    
        fig, axs = plt.subplots(nrows=1, ncols=2)
        
        axs[0].hist(x_MAE , density=True, histtype='bar', color=['red', 'green'], label=['Treinamento', 'Validação'])
        x, p = define_normal_distribution(axs[0], x_MAE[0])
        axs[0].plot(x, p, 'red', linewidth=2)
        x, p = define_normal_distribution(axs[0], x_MAE[1])
        axs[0].plot(x, p, 'green', linewidth=2)
        axs[0].set_title('MAE')
        
        axs[1].hist(x_RMSE, density=True, histtype='bar', color=['red', 'green'], label=['Treinamento', 'Validação'])
        x, p = define_normal_distribution(axs[1], x_RMSE[0])
        axs[1].plot(x, p, 'red', linewidth=2)
        x, p = define_normal_distribution(axs[1], x_RMSE[1])
        axs[1].plot(x, p, 'green', linewidth=2)
        axs[1].set_title('RMSE')
        
        for ax in axs.flat:
            ax.set(ylabel='Frequency')
    
        plt.suptitle(f'Histograms of model {model_name}')
        fig.tight_layout()
        
        if(showImages):
            plt.show()
        
        saveFig(plt, 'Histograms.', model_name, model_name)
        plt.close()

def showResidualPlots(true_values_dict, predicted_values_dict, city_cluster_name, city_for_training, city_for_predicting, showImages):
    residuals        = {'Train': true_values_dict['Train'] - predicted_values_dict['Train'],
                        'Test' : true_values_dict['Test' ] - predicted_values_dict['Test' ]}
    
    for training_or_testing in ['Train', 'Test']:
        plt.scatter(predicted_values_dict[training_or_testing], residuals[training_or_testing], alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot for {training_or_testing} data. Model {city_for_training} applied to {city_for_predicting}.')
        if(showImages):
            plt.show()
        
        saveFig(plt, f'Residual Plots {training_or_testing}', city_cluster_name, city_for_training, city_for_predicting)
        plt.close()

def showR2ScatterPlots(true_values_dict, predicted_values_dict, city_cluster_name, city_for_training, city_for_predicting, showImages):    
    for training_or_testing in ['Train', 'Test']:
        plt.scatter(true_values_dict[training_or_testing], predicted_values_dict[training_or_testing], label = 'R²')
        
        # Generates a single line by creating `x_vals`, a sequence of 100 evenly spaced values between the min and max values in true_values
        flattened_values = np.ravel(true_values_dict[training_or_testing])
        x_vals = np.linspace(min(flattened_values), max(flattened_values), 100)
        plt.plot(x_vals, x_vals, color='red', label='x=y')  # Line will only appear once
        
        plt.title(f'R² {training_or_testing} data. Model {city_for_training} applied to {city_for_predicting}.')
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
        plt.legend()
            
        if(showImages):
            plt.show()
        
        saveFig(plt, f'R² Scatter Plot {training_or_testing}', city_cluster_name, city_for_training, city_for_predicting)
        plt.close()

def drawMetricsRadarPlots(metrics_df, showImages):
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
    for metric_type in list_of_metrics_types:
        for model_name in list_of_models_names:
            values     = [ metrics_averages_dict['MAE' ][metric_type][model_name],
                           metrics_averages_dict['RMSE'][metric_type][model_name],
                           metrics_averages_dict['MSE' ][metric_type][model_name],
                           metrics_averages_dict['R^2' ][metric_type][model_name] ]
            
            # Compute angle for each category:
            angles = np.linspace(0, 2 * np.pi, len(list_of_metrics_names), endpoint=False).tolist() + [0]
            
            plt.polar (angles, values + values[:1], color='red', linewidth=1)
            plt.fill  (angles, values + values[:1], color='red', alpha=0.25)
            plt.xticks(angles[:-1], list_of_metrics_names)
            
            # To prevent the radial labels from overlapping:
            ax = plt.subplot(111, polar=True)
            ax.set_theta_offset(np.pi / 2)   # Set the offset
            ax.set_theta_direction(-1)       # Set direction to clockwise
    
            
            plt.title (f'Performance of model {model_name} ({metric_type})')
            plt.tight_layout()
            
            if(showImages):
                plt.show()
                
            saveFig(plt, f'Radar Plots. {model_name}. {metric_name}. {metric_type}.', model_name, model_name)
            plt.close()

def showTaylorDiagrams(errors_dict, SPEI_dict, dataTrueValues_dict, predictValues_dict, city_cluster_name, city_for_training, city_for_predicting, showImages):
    # Calculate precision measures:
    ## Standard Deviation:
    train_predictions_std_dev = np.std(predictValues_dict['Train'])
    test_predictions_std_dev  = np.std(predictValues_dict['Test' ])
    
    combined_data             = np.concatenate([SPEI_dict['Train'], SPEI_dict['Test']])
    observed_std_dev          = np.std(combined_data)
    
    print(f'\t\t\tTRAIN: STD Dev {train_predictions_std_dev}')
    print(f'\t\t\tTEST : STD Dev {test_predictions_std_dev }')
    
    ## Correlation Coefficient:
    train_data_model_corr = np.corrcoef(predictValues_dict['Train'], dataTrueValues_dict['Train'])[0, 1]
    test_data_model_corr  = np.corrcoef(predictValues_dict['Test' ], dataTrueValues_dict['Test' ])[0, 1]
    
    print(f'\t\t\tTRAIN: correlation {train_data_model_corr}')
    print(f'\t\t\tTEST : correlation {test_data_model_corr}' )
    
    label =          [      'Obs'     ,          'Train'            ,           'Test'           ]
    sdev  = np.array([observed_std_dev, train_predictions_std_dev   , test_predictions_std_dev   ])
    ccoef = np.array([       1.       , train_data_model_corr       , test_data_model_corr       ])
    rmse  = np.array([       0.       , errors_dict['Train']['RMSE'], errors_dict['Test']['RMSE']])
    
    # Plotting:
    ## If both are positive, 90° (2 squares), if one of them is negative, 180° (2 rectangles)
    figsize = (2*8, 2*5) if (train_data_model_corr > 0 and test_data_model_corr > 0) else (2*8, 2*3)
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)
    AVAILABLE_AXES = {'a) Training': 0, 'b) Testing': 1}
    for axs_title, axs_number in AVAILABLE_AXES.items():
        ax = axs[axs_number]
        ax.set_title(axs_title, loc="left", y=1.1)
        ax.set(adjustable='box', aspect='equal')
        sm.taylor_diagram(ax, sdev, rmse, ccoef, markerLabel = label, markerLabelColor = 'r', 
                          markerLegend = 'on', markerColor = 'r',
                          styleOBS = '-', colOBS = 'r', markerobs = 'o',
                          markerSize = 6, tickRMS = [0.0, 0.05, 0.1, 0.15, 0.2],
                          tickRMSangle = 115, showlabelsRMS = 'on',
                          titleRMS = 'on', titleOBS = 'Obs')
    plt.suptitle (f'Model {city_for_training} applied to {city_for_predicting}')
    fig.tight_layout()
    
    if(showImages):
        plt.show()
        
    saveFig(plt, 'Taylor Diagram.', city_cluster_name, city_for_training, city_for_predicting)
    plt.close()