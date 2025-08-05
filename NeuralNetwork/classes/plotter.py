# import statistics                      # Unused for now
import os
import matplotlib.pyplot   as      plt
import numpy               as       np
# import skill_metrics       as       sm # Unused for now
# import matplotlib.gridspec as gridspec # Unused for now
# from   scipy.stats         import norm # Unused for now

class Plotter:
    
    METRICS_PORTIONS_CENTRAL   = [ '80%', '20%']
    METRICS_PORTIONS_BORDERING = ['100%', '20%']
    
    def __init__(self, dataset):
        self.dataset              = dataset
        self.monthValues          = self.dataset.get_months          ()
        self.speiValues           = self.dataset.get_spei            ()
        self.speiNormalizedValues = self.dataset.get_spei_normalized ()

    def _saveFig(self, plot, filename, city_cluster_name=None, city_for_training=None, city_for_predicting=None):
        if city_for_predicting:
            FILEPATH = f'./Images/cluster {city_cluster_name}/model {city_for_training}/city {city_for_predicting}/'
            os.makedirs(FILEPATH, exist_ok=True)
            plt.savefig(FILEPATH + filename + f' - Model {city_for_training} applied to {city_for_predicting}.png')
        elif city_for_training:
            FILEPATH = f'./Images/cluster {city_cluster_name}/model {city_for_training}/'
            os.makedirs(FILEPATH, exist_ok=True)
            plt.savefig(FILEPATH + filename + f' - Model {city_for_training}.png')
        else:
            FILEPATH = './Images/'
            os.makedirs(FILEPATH, exist_ok=True)
            plt.savefig(FILEPATH + filename, bbox_inches="tight")

    def plotDatasetPlots(self, dataset, spei_test, split, city_cluster_name, city_for_training, city_for_predicting):
        self.showSpeiData(dataset     , spei_test, split, city_cluster_name, city_for_training, city_for_predicting)
        self.showSpeiTest(dataset     , spei_test, split, city_cluster_name, city_for_training, city_for_predicting)

    def plotModelPlots(self                  , spei_dict            , is_model           ,
                       spei_expected_outputs , spei_predicted_values,
                       monthForPredicted_dict, has_trained          ,
                       history               , metrics_df           ,
                       city_cluster_name     , city_for_training    , city_for_predicting):
        
        # Issue #7: "Taylor Diagrams are an unfinished work":
        # https://github.com/A-Infor/Python-OOP-LSTM-Drought-Predictor/issues/7
        # self.showTaylorDiagrams         (metrics_df                             , city_cluster_name, city_for_training, city_for_predicting)
        
        self.showResidualPlots           (is_model         , spei_expected_outputs, spei_predicted_values,
                                          city_cluster_name, city_for_training    , city_for_predicting  )
        self.showR2ScatterPlots          (is_model         , spei_expected_outputs, spei_predicted_values,
                                          city_cluster_name, city_for_training    , city_for_predicting  )
        self.showPredictionsDistribution (is_model         , spei_expected_outputs, spei_predicted_values,
                                          city_cluster_name, city_for_training    , city_for_predicting  )
        self.showPredictionResults       (is_model         , spei_expected_outputs, spei_predicted_values , monthForPredicted_dict,
                                          city_cluster_name, city_for_training   , city_for_predicting)
    
    # Disabled, as these are not going to be used on Anderson's masters dissertation:
    # def plotMetricsPlots(self, metrics_df):
    #     self.drawMetricsBoxPlots   (metrics_df)
    #     self.drawMetricsBarPlots   (metrics_df)
    #     self.drawMetricsHistograms (metrics_df)
        
    #     # Issue #3: "Radar Plots are an unfinished work"
    #     # https://github.com/A-Infor/Python-OOP-LSTM-Drought-Predictor/issues/3
    #     # self.drawMetricsRadarPlots (metrics_df)
    
    def showSpeiData(self, dataset, spei_test, split, city_cluster_name, city_for_training, city_for_predicting):
        monthValues          = dataset.get_months         ()
        speiValues           = dataset.get_spei           ()
        speiNormalizedValues = dataset.get_spei_normalized()
        
        plt.figure ()
        plt.subplot(2,1,1)
        plt.plot   (monthValues, speiValues          , label='SPEI Original'         )
        plt.xlabel ('Ano')
        plt.ylabel ('SPEI')
        plt.title  (f'SPEI Data - {city_for_predicting}')
        plt.legend ()
    
        plt.subplot(2,1,2)
        plt.plot   (monthValues, speiNormalizedValues, label='80%')
        plt.xlabel ('Ano')
        plt.ylabel ('SPEI (Normalizado)')
        plt.plot   (monthValues[split:],spei_test,'k',label='20%')
        plt.legend ()
        #plt.show()
        
        self._saveFig(plt, 'SPEI Data', city_cluster_name, city_for_training, city_for_predicting)
        plt.close()
    
    def showSpeiTest(self, dataset, spei_test, split, city_cluster_name, city_for_training, city_for_predicting):
        monthValues          = dataset.get_months()
        speiValues           = dataset.get_spei  ()
        
        y1positive = np.array(self.speiValues)>=0
        y1negative = np.array(self.speiValues)<=0
    
        plt.figure()
        plt.fill_between(monthValues, speiValues, y2=0, where=y1positive,
                         color='green', alpha=0.5, interpolate=False, label='índices SPEI positivos')
        plt.fill_between(monthValues, speiValues, y2=0, where=y1negative,
                         color='red'  , alpha=0.5, interpolate=False, label='índices SPEI negativos')
        plt.xlabel      ('Ano')
        plt.ylabel      ('SPEI')        
        plt.title       (f'{city_for_predicting}: SPEI Data')
        plt.legend      ()
        #plt.show()
        
        self._saveFig(plt, 'SPEI Data (test)', city_cluster_name, city_for_training, city_for_predicting)
        plt.close()
    
    def _calculateDenormalizedValues(self, is_model, spei_expected_outputs, spei_predicted_values):
        
        ###ADJUSTMENTS OF INPUTS###############################################
        spei_expected_outputs['100%']  = spei_expected_outputs['100%'].flatten()
        spei_expected_outputs[ '20%']  = spei_expected_outputs[ '20%'].flatten()
        
        if is_model: spei_predicted_values['100%'] = np.append(spei_predicted_values[ '80%'],
                                                               spei_predicted_values[ '20%'])
        else       : spei_predicted_values['100%'] =           spei_predicted_values['100%'].flatten()
        
        ###PREPARATIVES FOR OUTPUT#############################################
        RELEVANT_PORTIONS             = ['100%', '20%']
        
        true_values_denormalized_dict = dict.fromkeys(RELEVANT_PORTIONS)
        predictions_denormalized_dict = dict.fromkeys(RELEVANT_PORTIONS)
        
        ###MIN & MAX FOR CALCULATION###########################################
        spei_max_value = np.max(self.speiValues)
        spei_min_value = np.min(self.speiValues)
        
        spei_delta     = spei_max_value - spei_min_value
        ###CALCULATIONS########################################################
        true_values_denormalized_dict['100%'] = (spei_expected_outputs ['100%']           * spei_delta + spei_min_value)
        true_values_denormalized_dict[ '20%'] = (spei_expected_outputs [ '20%']           * spei_delta + spei_min_value)
        
        predictions_denormalized_dict['100%'] = (spei_predicted_values['100%']           * spei_delta + spei_min_value)
        predictions_denormalized_dict[ '20%'] = (spei_predicted_values[ '20%'].flatten() * spei_delta + spei_min_value)
        
        return true_values_denormalized_dict, predictions_denormalized_dict
    
    def showPredictionResults(self      ,    is_model   , spei_expected_outputs, spei_predicted_values,
                              months_for_expected_outputs, city_cluster_name   , city_for_training    , city_for_predicting):
        
        (trueValues_denormalized ,
         predictions_denormalized) = self._calculateDenormalizedValues(is_model, spei_expected_outputs, spei_predicted_values)
        ###100%################################################################
        reshapedMonth = np.append(months_for_expected_outputs['80%'], months_for_expected_outputs['20%'])
    
        plt.figure ()
        plt.plot   (reshapedMonth,  trueValues_denormalized['100%'])
        plt.plot   (reshapedMonth, predictions_denormalized['100%'])
        plt.axvline(months_for_expected_outputs['80%'][-1][-1], color='r')
        plt.legend (['Real', 'Predicted'])
        plt.xlabel ('Year')
        plt.ylabel ('SPEI')
        plt.title  (f'Model {city_for_training} applied to {city_for_predicting}:\nreal and predicted SPEI values (100%\'s)')
        #plt.show()
        
        self._saveFig(plt, 'Previsao 100%', city_cluster_name, city_for_training, city_for_predicting)
        plt.close()
        ###20%#################################################################
        reshapedMonth = months_for_expected_outputs['20%'].flatten()
    
        plt.figure ()
        # ValueError: x and y can be no greater than 2D, but have shapes (11, 6, 1) and (11, 6):
        plt.plot   (reshapedMonth,  trueValues_denormalized[ '20%'])
        plt.plot   (reshapedMonth, predictions_denormalized[ '20%'])
        plt.legend (['Real', 'Predicted'])
        plt.xlabel ('Year')
        plt.ylabel ('SPEI')
        plt.title  (f'Model {city_for_training} applied to {city_for_predicting}:\nreal and predicted SPEI values (20%\'s)')
        #plt.show()
        
        self._saveFig(plt, 'Previsao 20%', city_cluster_name, city_for_training, city_for_predicting)
        plt.close()
        #######################################################################
    
    def showPredictionsDistribution(self, is_model   , spei_expected_outputs, spei_predicted_values,
                                    city_cluster_name, city_for_training   , city_for_predicting  ):
        
        (trueValues_denormalized ,
         predictions_denormalized) = self._calculateDenormalizedValues(is_model, spei_expected_outputs, spei_predicted_values)
        ###100%################################################################
        plt.figure ()
        plt.scatter(x =  trueValues_denormalized['100%'],
                    y = predictions_denormalized['100%'],
                    color=['white'],  marker='^', edgecolors='black')
        plt.xlabel ('Real SPEI')
        plt.ylabel ('Predicted SPEI'  )
        plt.axline ( (0, 0) , slope=1 )
        plt.title  (f'Model {city_for_training} applied to {city_for_predicting}:\nSPEI (100%\'s distribution)')
        #plt.show()
        
        self._saveFig(plt, 'distribuiçãoDoSPEI 100%', city_cluster_name, city_for_training, city_for_predicting)
        plt.close()
        ###20%#################################################################
        plt.figure ()
        plt.scatter(x =  trueValues_denormalized[ '20%'],
                    y = predictions_denormalized[ '20%'],
                    color=['white'],  marker='D', edgecolors='black')
        plt.xlabel ('Real SPEI')
        plt.ylabel ('Predicted SPEI'  )
        plt.axline ( (0, 0) , slope=1 )
        plt.title  (f'Model {city_for_training} applied to {city_for_predicting}:\nSPEI (20%\'s distribution)')
        #plt.show()
        
        self._saveFig(plt, 'distribuiçãoDoSPEI 20%', city_cluster_name, city_for_training, city_for_predicting)
        plt.close()
        #######################################################################

    def drawModelLineGraph(self, history, city_cluster_name, city_for_training):
        
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
    
        self._saveFig(plt, 'Line Graph.', city_cluster_name, city_for_training)
        plt.close()

    def define_box_properties(self, plot_name, color_code, label):
        	for k, v in plot_name.items():
        		plt.setp(plot_name.get(k), color=color_code)
        		
        	# use plot function to draw a small line to name the legend.
        	plt.plot([], c=color_code, label=label)
        	plt.legend()
    
    def showResidualPlots(self  ,  is_model, true_values_dict , predicted_values_dict,
                          city_cluster_name, city_for_training, city_for_predicting  ):
        
        if is_model:
            residuals        = { '80%': true_values_dict[ '80%'] - predicted_values_dict[ '80%'],
                                 '20%': true_values_dict[ '20%'] - predicted_values_dict[ '20%']}
        else:
            residuals        = {'100%': true_values_dict['100%'] - predicted_values_dict['100%'],
                                 '20%': true_values_dict[ '20%'] - predicted_values_dict[ '20%']}
        
        for data_portion_type in Plotter.METRICS_PORTIONS_CENTRAL if is_model else Plotter.METRICS_PORTIONS_BORDERING:
            plt.scatter(predicted_values_dict[data_portion_type], residuals[data_portion_type], alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title (f'Residual Plot for {data_portion_type} data.\nModel {city_for_training} applied to {city_for_predicting}.')
            
            self._saveFig(plt, f'Residual Plots {data_portion_type}', city_cluster_name, city_for_training, city_for_predicting)
            plt.close()
    
    def showR2ScatterPlots(self, is_model, true_values_dict, predicted_values_dict, city_cluster_name, city_for_training, city_for_predicting):
        for data_portion_type in Plotter.METRICS_PORTIONS_CENTRAL if is_model else Plotter.METRICS_PORTIONS_BORDERING:
            plt.scatter(true_values_dict[data_portion_type], predicted_values_dict[data_portion_type], label = 'R²')
            
            # Generates a single line by creating `x_vals`, a sequence of 100 evenly spaced values between the min and max values in true_values
            flattened_values = np.ravel(true_values_dict[data_portion_type])
            x_vals = np.linspace(min(flattened_values), max(flattened_values), 100)
            plt.plot(x_vals, x_vals, color='red', label='x=y')  # Line will only appear once
            
            plt.title (f'R² {data_portion_type} data. Model {city_for_training} applied to {city_for_predicting}.')
            plt.xlabel('True values')
            plt.ylabel('Predicted values')
            plt.legend()
                
            self._saveFig(plt, f'R² Scatter Plot {data_portion_type}', city_cluster_name, city_for_training, city_for_predicting)
            plt.close()

    # Disabled, as these are not going to be used on Anderson's masters dissertation:
    # def drawMetricsBoxPlots(self, metrics_df):   
    #     # Creation of the empty dictionary:
    #     list_of_metrics_names = ['MAE', 'RMSE',    'MSE'   ]
    #     list_of_metrics_types = ['80%', '20%']
    #     list_of_models_names  = metrics_df['Municipio Treinado'].unique()
        
    #     metrics_dict = dict.fromkeys(list_of_metrics_names)
    #     for metric_name in metrics_dict.keys():
    #         metrics_dict[metric_name] = dict.fromkeys(list_of_metrics_types)
            
    #         for metric_type in metrics_dict[metric_name].keys():
    #             metrics_dict[metric_name][metric_type] = dict.fromkeys(list_of_models_names)
        
    #     # Filling the dictionary:
    #     for metric_name in list_of_metrics_names:
    #         for metric_type in list_of_metrics_types:
    #             for model_name in list_of_models_names:
    #                 df_filter = metrics_df['Municipio Treinado'] == model_name
    #                 metrics_dict[metric_name][metric_type][model_name] = metrics_df[ df_filter ][f'{metric_name} {metric_type}'].to_list()
        
    #     # Plotting the graphs:
    #     for metric_name in list_of_metrics_names:
    #         training_values   = metrics_dict[metric_name]['80%'].values()
    #         boxplot_values_positions_base = np.array(np.arange(len(training_values  )))
    #         training_plot     = plt.boxplot(training_values  , positions=boxplot_values_positions_base*2.0-0.35)
            
    #         testing_values = metrics_dict[metric_name]['20%'  ].values()
    #         testing_plot   = plt.boxplot(testing_values, positions=boxplot_values_positions_base*2.0+0.35)
        
    #         # setting colors for each groups
    #         self.define_box_properties(training_plot  , '#D7191C', '80%'  )
    #         self.define_box_properties(testing_plot, '#2C7BB6', '20%')
        
    #         # set the x label values
    #         testing_keys = metrics_dict[metric_name]['20%'].keys()
    #         plt.xticks(np.arange(0, len(testing_keys) * 2, 2), testing_keys, rotation=45)
            
    #         plt.title (f'Comparison of performance of different models ({metric_name})')
    #         plt.xlabel('Machine Learning models')
    #         plt.ylabel(f'{metric_name} values')
    #         plt.grid  (axis='y', linestyle=':', color='gray', linewidth=0.7)
            
    #         self._saveFig(plt, f'Box Plots. {metric_name}.')
    #         plt.close()               
        
    
    # Disabled, as these are not going to be used on Anderson's masters dissertation:
    # def drawMetricsBarPlots(self, metrics_df):
    #     # Creation of the empty dictionary:
    #     list_of_metrics_names = ['MAE', 'RMSE', 'MSE', 'R^2']
    #     list_of_metrics_types = ['80%', '20%' ]
    #     list_of_models_names  = metrics_df['Municipio Treinado'].unique()
        
    #     metrics_averages_dict = dict.fromkeys(list_of_metrics_names)
    #     for metric_name in metrics_averages_dict.keys():
    #         metrics_averages_dict[metric_name] = dict.fromkeys(list_of_metrics_types)
            
    #         for metric_type in metrics_averages_dict[metric_name].keys():
    #             metrics_averages_dict[metric_name][metric_type] = dict.fromkeys(list_of_models_names)
        
    #     # Filling the dictionary:
    #     for metric_name in list_of_metrics_names:
    #         for metric_type in list_of_metrics_types:
    #             for model_name in list_of_models_names:
    #                 df_filter = metrics_df['Municipio Treinado'] == model_name
    #                 average = statistics.mean( metrics_df[ df_filter ][f'{metric_name} {metric_type}'].to_list() )
    #                 metrics_averages_dict[metric_name][metric_type][model_name] = average
        
    #     # Plotting the graphs:
    #     for metric_name in list_of_metrics_names:
    #         Y_axis = np.arange(len(list_of_models_names)) 
            
    #         # 0.4: width of the bars; 0.2: distance between the groups
    #         plt.barh(Y_axis - 0.2, metrics_averages_dict[metric_name]['80%'].values(), 0.4, label = '80%'  )
    #         plt.barh(Y_axis + 0.2, metrics_averages_dict[metric_name]['20%'  ].values(), 0.4, label = '20%')
            
    #         plt.yticks(Y_axis, list_of_models_names, rotation=45)
    #         plt.ylabel("Machine Learning models")
    #         plt.xlabel(f'Average {metric_name}' if metric_name != 'R^2' else 'Average R²')
    #         plt.title ("Comparison of performance of different models")
    #         plt.legend()
            
    #         self._saveFig(plt, f'Bar Plots. {metric_name}.')
    #         plt.close()
    
    # Disabled, as these are not going to be used on Anderson's masters dissertation:
    # def define_normal_distribution(self, axis, x_values):
    #     if np.std(x_values) > 0:
    #         mu  , std  = norm.fit     (x_values)
    #         xmin, xmax = axis.get_xlim()
    #         x          = np.linspace  (xmin, xmax, 100)
    #         p          = norm.pdf     (x   , mu  , std)
            
    #         return x, p
    #     else:
    #         print('Info: normal distribution <= 0')
    #         return 0, 0
    
    # Disabled, as these are not going to be used on Anderson's masters dissertation:
    # def drawMetricsHistograms(self, metrics_df):
    #     COLS_LABELS = ['80% (columns)', '20% (columns)']
    #     COLS_COLORS = [        'red'       ,        'green'        ]
        
    #     # Creation of the empty dictionary:
    #     list_of_metrics_names = [    'MAE'    ,   'RMSE'   ]
    #     list_of_metrics_types = ['80%', '20%']
    #     list_of_models_names  = metrics_df['Municipio Treinado'].unique()
        
    #     metrics_dict = dict.fromkeys(list_of_metrics_names)
    #     for metric_name in metrics_dict.keys():
    #         metrics_dict[metric_name] = dict.fromkeys(list_of_metrics_types)
            
    #         for metric_type in metrics_dict[metric_name].keys():
    #             metrics_dict[metric_name][metric_type] = dict.fromkeys(list_of_models_names)
        
    #     # Filling the dictionary:
    #     for metric_name in list_of_metrics_names:
    #         for metric_type in list_of_metrics_types:
    #             for model_name in list_of_models_names:
    #                 df_filter = metrics_df['Municipio Treinado'] == model_name
    #                 metrics_dict[metric_name][metric_type][model_name] = metrics_df[ df_filter ][f'{metric_name} {metric_type}'].to_list()
    
    #     # Plotting the graphs:
    #     for model_name in list_of_models_names:
    #         x_MAE  = [ metrics_dict['MAE' ]['80%'][model_name] ,
    #                    metrics_dict['MAE' ]['20%'][model_name] ]
    #         x_RMSE = [ metrics_dict['RMSE']['80%'][model_name] ,
    #                    metrics_dict['RMSE']['20%'][model_name] ]
            
    #         # fig = plt.figure(figsize=(12, 8))
    #         fig = plt.figure()
    #         gs  = gridspec.GridSpec(2, 2, height_ratios=[10, 1])

    #         # LAYOUT:
    #         # +------------------------+------------------------+
    #         # |           MAE          |          RMSE          |
    #         # +------------------------+------------------------+
    #         # |                      LEGEND                     |
    #         # +-------------------------------------------------+

    #         ax_mae    = fig.add_subplot(gs[0, 0])
    #         ax_rmse   = fig.add_subplot(gs[0, 1])
    #         ax_legend = fig.add_subplot(gs[1, :])
            
    #         # MAE Histogram
    #         ax_mae.hist(x_MAE[0], bins='auto', histtype='bar', color=COLS_COLORS[0], label=COLS_LABELS[0], alpha=0.6, density=False)
    #         ax_mae.hist(x_MAE[1], bins='auto', histtype='bar', color=COLS_COLORS[1], label=COLS_LABELS[1], alpha=0.6, density=False)
    #         x, p = self.define_normal_distribution(ax_mae, x_MAE[0])
    #         ax_mae.plot(x, p, 'red', linewidth=2, label='80% Normal Distribution (curves)')
    #         x, p = self.define_normal_distribution(ax_mae, x_MAE[1])
    #         ax_mae.plot(x, p, 'green', linewidth=2, label='20% Normal Distribution (curves)')
    #         ax_mae.set_title('MAE')
    #         ax_mae.set_ylabel('Frequency')
            
    #         # RMSE Histogram
    #         ax_rmse.hist(x_RMSE[0], bins='auto', histtype='bar', color=COLS_COLORS[0], label=COLS_LABELS[0], alpha=0.6, density=False)
    #         ax_rmse.hist(x_RMSE[1], bins='auto', histtype='bar', color=COLS_COLORS[1], label=COLS_LABELS[1], alpha=0.6, density=False)
    #         x, p = self.define_normal_distribution(ax_rmse, x_RMSE[0])
    #         ax_rmse.plot(x, p, 'red', linewidth=2, label='80% Normal Distribution (curves)')
    #         x, p = self.define_normal_distribution(ax_rmse, x_RMSE[1])
    #         ax_rmse.plot(x, p, 'green', linewidth=2, label='20% Normal Distribution (curves)')
    #         ax_rmse.set_title('RMSE')
            
    #         # Plot legend in separate subplot
    #         ax_legend.axis('off')
    #         handles, labels = ax_mae .get_legend_handles_labels()
    #         ax_legend.legend(
    #             handles, labels,
    #             loc='center', ncol=2, frameon=False
    #         )
            
    #         fig.suptitle(f'Histograms of model {model_name}')
    #         fig.tight_layout(rect=[0, 0, 1, 0.95])
            
    #         self._saveFig(plt, 'Histograms.', model_name, model_name)
    #         plt.close()

    # Issue #3: "Radar Plots are an unfinished work"
    # https://github.com/A-Infor/Python-OOP-LSTM-Drought-Predictor/issues/3    
    #
    # def drawMetricsRadarPlots(self, metrics_df):
    #     # Creation of the empty dictionary:
    #     list_of_metrics_names = ['MAE', 'RMSE', 'MSE', 'R^2']
    #     list_of_metrics_types = ['80%', '20%']
    #     list_of_models_names  = metrics_df['Municipio Treinado'].unique()
        
    #     metrics_averages_dict = dict.fromkeys(list_of_metrics_names)
    #     for metric_name in metrics_averages_dict.keys():
    #         metrics_averages_dict[metric_name] = dict.fromkeys(list_of_metrics_types)
            
    #         for metric_type in metrics_averages_dict[metric_name].keys():
    #             metrics_averages_dict[metric_name][metric_type] = dict.fromkeys(list_of_models_names)
        
    #     # Filling the dictionary:
    #     for metric_name in list_of_metrics_names:
    #         for metric_type in list_of_metrics_types:
    #             for model_name in list_of_models_names:
    #                 df_filter = metrics_df['Municipio Treinado'] == model_name
    #                 average = statistics.mean( metrics_df[ df_filter ][f'{metric_name} {metric_type}'].to_list() )
    #                 metrics_averages_dict[metric_name][metric_type][model_name] = average
        
    #     # Plotting the graphs:
    #     for metric_type in list_of_metrics_types:
    #         for model_name in list_of_models_names:
    #             values     = [ metrics_averages_dict['MAE' ][metric_type][model_name],
    #                            metrics_averages_dict['RMSE'][metric_type][model_name],
    #                            metrics_averages_dict['MSE' ][metric_type][model_name],
    #                            metrics_averages_dict['R^2' ][metric_type][model_name] ]
                
    #             # Compute angle for each category:
    #             angles = np.linspace(0, 2 * np.pi, len(list_of_metrics_names), endpoint=False).tolist() + [0]
                
    #             plt.polar (angles, values + values[:1], color='red', linewidth=1)
    #             plt.fill  (angles, values + values[:1], color='red', alpha=0.25)
    #             plt.xticks(angles[:-1], list_of_metrics_names)
                
    #             # To prevent the radial labels from overlapping:
    #             ax = plt.subplot(111, polar=True)
    #             ax.set_theta_offset(np.pi / 2)   # Set the offset
    #             ax.set_theta_direction(-1)       # Set direction to clockwise
        
                
    #             plt.title (f'Performance of model {model_name} ({metric_type})')
    #             plt.tight_layout()
                
    #             self._saveFig(plt, f'Radar Plots. {model_name}. {metric_name}. {metric_type}.', model_name, model_name)
    #             plt.close()


    # Issue #7: "Taylor Diagrams are an unfinished work":
    # https://github.com/A-Infor/Python-OOP-LSTM-Drought-Predictor/issues/7    
    # 
    # def showTaylorDiagrams(self, metrics_df, city_cluster_name, city_for_training, city_for_predicting):
        
    #     label =          ['Obs', '80%', '20%']
    #     sdev  = np.array([metrics_df.iloc[-1]['Desvio Padrão Obs.'             ] ,
    #                       metrics_df.iloc[-1]['Desvio Padrão Pred. 80%'] ,
    #                       metrics_df.iloc[-1]['Desvio Padrão Pred. 20%'  ] ])
    #     ccoef = np.array([1.                                                     ,
    #                       metrics_df.iloc[-1]['Coef. de Correlação 80%'] ,
    #                       metrics_df.iloc[-1]['Coef. de Correlação 20%'  ] ])
    #     rmse  = np.array([0.                                                     ,
    #                       metrics_df.iloc[-1]['RMSE 80%'               ] ,
    #                       metrics_df.iloc[-1]['RMSE 20%'                 ] ])
        
    #     # Plotting:
    #     ## If both are positive, 90° (2 squares), if one of them is negative, 180° (2 rectangles)
    #     figsize = (2*8, 2*5) if (metrics_df.iloc[-1]['Coef. de Correlação 80%'] > 0 and metrics_df.iloc[-1]['Coef. de Correlação 20%'] > 0) else (2*8, 2*3)
        
    #     fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)
    #     AVAILABLE_AXES = {'a) 80%': 0, 'b) Testing': 1}
    #     for axs_title, axs_number in AVAILABLE_AXES.items():
    #         ax = axs[axs_number]
    #         ax.set_title(axs_title, loc="left", y=1.1)
    #         ax.set(adjustable='box', aspect='equal')
    #         sm.taylor_diagram(ax, sdev, rmse, ccoef, markerLabel = label, markerLabelColor = 'r', 
    #                           markerLegend = 'on', markerColor   = 'r' ,
    #                           styleOBS     = '-' , colOBS        = 'r' ,       markerobs = 'o',
    #                           markerSize   =   6 , tickRMS       = [0.0, 0.05, 0.1, 0.15, 0.2],
    #                           tickRMSangle = 115 , showlabelsRMS = 'on',
    #                           titleRMS     = 'on', titleOBS      = 'Obs')
    #     plt.suptitle (f'Model {city_for_training} applied to {city_for_predicting}')
    #     fig.tight_layout(pad = 1.5)
        
    #     self._saveFig(plt, 'Taylor Diagram.', city_cluster_name, city_for_training, city_for_predicting)
    #     plt.close()