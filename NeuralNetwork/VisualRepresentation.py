import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.stats import norm

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

def showResidualPlots(training_true_values, training_predicted_values, testing_true_values, testing_predicted_values, city_cluster_name, city_for_training, city_for_predicting, showImages):
    predicted_values = {'Training': training_predicted_values,
                        'Testing' :  testing_predicted_values}
    
    residuals        = {'Training': training_true_values - predicted_values['Training'],
                        'Testing' :  testing_true_values - predicted_values['Testing' ]}
    
    for training_or_testing, residual_values in residuals.items():
        plt.scatter(predicted_values[training_or_testing], residuals[training_or_testing], alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot for {training_or_testing} data. Model {city_for_training} applied to {city_for_predicting}.')
        if(showImages):
            plt.show()
        
        saveFig(plt, f'Residual Plots {training_or_testing}', city_cluster_name, city_for_training, city_for_predicting)
        plt.close()

def showR2ScatterPlots(training_true_values, training_predicted_values, testing_true_values, testing_predicted_values, city_cluster_name, city_for_training, city_for_predicting, showImages):    
    true_values      = {'Training': training_true_values,
                        'Testing' :  testing_true_values}
    
    predicted_values = {'Training': training_predicted_values,
                        'Testing' :  testing_predicted_values}
    
    for training_or_testing in ['Training', 'Testing']:
        plt.scatter(true_values[training_or_testing], predicted_values[training_or_testing], label = 'R²')
        
        # Generates a single line by creating `x_vals`, a sequence of 100 evenly spaced values between the min and max values in true_values
        flattened_values = np.ravel(true_values[training_or_testing])
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

def showRMSETaylorDiagrams(training_true_values, training_predicted_values, testing_true_values, testing_predicted_values, city_cluster_name, city_for_training, city_for_predicting, showImages):
    # Calculate precision measures:
    ## Standard Deviation:
    train_predictions_std_dev = np.std(training_predicted_values)
    test_predictions_std_dev  = np.std(testing_predicted_values )
    
    print(f'\t\t\tTRAIN: STD Dev {train_predictions_std_dev}')
    print(f'\t\t\tTEST : STD Dev {test_predictions_std_dev }')
    
    ## Correlation Coefficient:
    train_data_model_corr = np.corrcoef(training_predicted_values, training_true_values)[0, 1]
    test_data_model_corr  = np.corrcoef(testing_predicted_values , testing_true_values )[0, 1]
    ### If both values are positive, the left part of the semicircle (negative values) isn't needed:
    degrees =  90 if (train_data_model_corr > 0) and (test_data_model_corr > 0) else 180
    
    print(f'\t\t\tTRAIN: correlation {train_data_model_corr}')
    print(f'\t\t\tTEST : correlation {test_data_model_corr}' )
    
    # Plots the graph: 
    std_ref = 1.0                                                           # Reference standard deviation
    models_std_dev = [train_predictions_std_dev, test_predictions_std_dev]  # Standard deviations of models
    models_corr    = [train_data_model_corr    , test_data_model_corr    ]  # Correlation coefficients of models
    
    ## Create an empty Taylor Diagram:
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_thetalim(thetamin=0, thetamax=degrees) # thetamax = 90° or 180°
    ax.set_theta_zero_location('E')               # Set 0 degrees at the right
    ax.set_theta_direction(1)                     # Counter-clockwise direction
    
    ### X axis (angular):
    ax.xaxis.grid     (color='blue' )
    correlation_range = np.arange( 0 if degrees == 90 else -1, 1.1, 0.1)  # min. is 0 if 90° or -1 if 180º (max. 1.1 is discarded, real max. is 1)
    ax.set_xticks     (np.arccos(correlation_range))                      # Set the ticks at calculated angles (converted by arccos)
    ax.set_xticklabels([f'{c:.1f}' for c in correlation_range], color='blue', fontsize=10 if degrees == 90 else 7)  # Use correlation values as labels
    
    ### Y axis (radial):
    ax.yaxis.grid(color='black')
    ax.set_ylim(min(models_std_dev) * 0.8, max(models_std_dev) * 1.2)   # 20% margin
    
    #### Set the y-ticks
    yticks = np.arange(0, max(models_std_dev) * 1.2, 0.1)
    ax.set_yticks(yticks)
    if degrees == 180: ax.yaxis.set_tick_params(labelright=True) # (counterintuitively) Adds labels to the left half of the plot

    ### Axis labels    :
    ax.set_xlabel('Standard Deviation')
    ax.xaxis.set_label_coords(0.5, 0.15 if degrees == 180 else -0.1)
    if   degrees ==  90: ax.set_ylabel('Standard Deviation', labelpad=4)
    ax.text(np.pi / (4.2 if degrees == 90 else 2), 0.26, 'Pearson Correlation Coefficient', ha='center', va='center', color='blue', rotation = -45 if degrees == 90 else 0)
    
    ax.set_title(f'Model {city_for_training} applied to {city_for_predicting}')
    
    ## Fill in the Taylor Diagram:
    ax.plot(np.arccos(models_corr), models_std_dev,  'ro', label='Models'   )
    ax.plot(    [0], [std_ref]    ,                  'mo', label='Reference')
    plt.legend()
    
    ax.text(np.arccos(train_data_model_corr), train_predictions_std_dev + 0.005, 'train', ha='center', va='bottom')
    ax.text(np.arccos(test_data_model_corr ), test_predictions_std_dev  + 0.005, 'test' , ha='center', va='bottom')
    
    fig.tight_layout()
    
    if(showImages):
        plt.show()
        
    saveFig(plt, 'Taylor Diagram.', city_cluster_name, city_for_training, city_for_predicting)
    plt.close()
