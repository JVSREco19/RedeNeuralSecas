import statistics
import os
import matplotlib.pyplot   as      plt
import numpy               as       np
import skill_metrics       as       sm
import matplotlib.gridspec as gridspec
from   scipy.stats         import norm

class Plotter:
    
    def __init__(self, dataset):
        self.dataset              = dataset
        self.monthValues          = self.dataset.get_months()
        self.speiValues           = self.dataset.get_spei()
        self.speiNormalizedValues = self.dataset.get_spei_normalized()

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
        self.showSpeiData(dataset, spei_test, split, city_cluster_name, city_for_training, city_for_predicting)
        self.showSpeiTest(dataset, spei_test, split, city_cluster_name, city_for_training, city_for_predicting)

    def plotModelPlots(self                  , spei_dict         ,
                       dataTrueValues_dict   , predictValues_dict,
                       monthForPredicted_dict, has_trained       ,
                       history                                   , metrics_df         ,
                       city_cluster_name     , city_for_training , city_for_predicting):
        
        # Issue #7: "Taylor Diagrams are an unfinished work"
        # self.showTaylorDiagrams         (metrics_df                             , city_cluster_name, city_for_training, city_for_predicting)
        self.showResidualPlots          (dataTrueValues_dict, predictValues_dict, city_cluster_name, city_for_training, city_for_predicting)
        self.showR2ScatterPlots         (dataTrueValues_dict, predictValues_dict, city_cluster_name, city_for_training, city_for_predicting)
        self.showPredictionsDistribution(dataTrueValues_dict, predictValues_dict, city_cluster_name, city_for_training, city_for_predicting)
        self.showPredictionResults      (dataTrueValues_dict, predictValues_dict, monthForPredicted_dict, city_cluster_name, city_for_training, city_for_predicting)
    
    def plotMetricsPlots(self, metrics_df):
        self.drawMetricsBoxPlots   (metrics_df)
        self.drawMetricsBarPlots   (metrics_df)
        self.drawMetricsHistograms (metrics_df)
        
        # Issue #3: "Radar Plots are an unfinished work"
        # self.drawMetricsRadarPlots (metrics_df)
    
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
        plt.plot   (monthValues, speiNormalizedValues, label='Parcela de Treinamento')
        plt.xlabel ('Ano')
        plt.ylabel ('SPEI (Normalizado)')
        plt.plot   (monthValues[split:],spei_test,'k',label='Parcela de Teste')
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
        
    def showPredictionResults(self, dataTrueValues_dict, predictValues_dict, monthsForPredicted_dict, city_cluster_name, city_for_training, city_for_predicting):
        trueValues  = np.append(dataTrueValues_dict['Train'], dataTrueValues_dict['Test'])
        predictions = np.append( predictValues_dict['Train'],  predictValues_dict['Test'])
    
        reshapedMonth = np.append(monthsForPredicted_dict['Train'], monthsForPredicted_dict['Test'])
    
        speiMaxValue = np.max(self.speiValues)
        speiMinValue = np.min(self.speiValues)
    
        trueValues_denormalized  = (trueValues  * (speiMaxValue - speiMinValue) + speiMinValue)
        predictions_denormalized = (predictions * (speiMaxValue - speiMinValue) + speiMinValue)
    
        plt.figure ()
        plt.plot   (reshapedMonth,  trueValues_denormalized)
        plt.plot   (reshapedMonth, predictions_denormalized)
        plt.axvline(monthsForPredicted_dict['Train'][-1][-1], color='r')
        plt.legend (['Verdadeiros', 'Previstos'])
        plt.xlabel ('Data')
        plt.ylabel ('SPEI')
        plt.title  (f'{city_for_predicting}:\nvalores verdadeiros e previstos para o final das séries')
        #plt.show()
        
        self._saveFig(plt, 'Previsao', city_cluster_name, city_for_training, city_for_predicting)
        plt.close()
    
    def showPredictionsDistribution(self, dataTrueValues_dict, predictValues_dict, city_cluster_name, city_for_training, city_for_predicting):
        trueValues  = np.append(dataTrueValues_dict['Train'], dataTrueValues_dict['Test'])
        predictions = np.append( predictValues_dict['Train'],  predictValues_dict['Test'])
    
        speiMaxValue = np.max(self.speiValues)
        speiMinValue = np.min(self.speiValues)
    
        trueValues_denormalized  = (trueValues  * (speiMaxValue - speiMinValue) + speiMinValue)
        predictions_denormalized = (predictions * (speiMaxValue - speiMinValue) + speiMinValue)
    
        plt.figure ()
        plt.scatter(x=trueValues_denormalized, y=predictions_denormalized, color=['white'], marker='^', edgecolors='black')
        plt.xlabel ('SPEI Verdadeiros')
        plt.ylabel ('SPEI Previstos')
        plt.axline ((0, 0), slope=1)
        plt.title  (f'{city_for_predicting}: SPEI (distribuição)')
        #plt.show()
        
        self._saveFig(plt, 'distribuiçãoDoSPEI', city_cluster_name, city_for_training, city_for_predicting)
        plt.close()

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
    
    def drawMetricsBoxPlots(self, metrics_df):   
        # Creation of the empty dictionary:
        list_of_metrics_names = ['MAE', 'RMSE',    'MSE'   ]
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
                    df_filter = metrics_df['Municipio Treinado'] == model_name
                    metrics_dict[metric_name][metric_type][model_name] = metrics_df[ df_filter ][f'{metric_name} {metric_type}'].to_list()
        
        # Plotting the graphs:
        for metric_name in list_of_metrics_names:
            training_values   = metrics_dict[metric_name]['Treinamento'].values()
            boxplot_values_positions_base = np.array(np.arange(len(training_values  )))
            training_plot     = plt.boxplot(training_values  , positions=boxplot_values_positions_base*2.0-0.35)
            
            validation_values = metrics_dict[metric_name]['Validação'  ].values()
            validation_plot   = plt.boxplot(validation_values, positions=boxplot_values_positions_base*2.0+0.35)
        
            # setting colors for each groups
            self.define_box_properties(training_plot  , '#D7191C', 'Training'  )
            self.define_box_properties(validation_plot, '#2C7BB6', 'Validation')
        
            # set the x label values
            validation_keys = metrics_dict[metric_name]['Validação'].keys()
            plt.xticks(np.arange(0, len(validation_keys) * 2, 2), validation_keys, rotation=45)
            
            plt.title (f'Comparison of performance of different models ({metric_name})')
            plt.xlabel('Machine Learning models')
            plt.ylabel(f'{metric_name} values')
            plt.grid  (axis='y', linestyle=':', color='gray', linewidth=0.7)
            
            self._saveFig(plt, f'Box Plots. {metric_name}.')
            plt.close()               
        
    
    def drawMetricsBarPlots(self, metrics_df):
        # Creation of the empty dictionary:
        list_of_metrics_names = ['MAE', 'RMSE', 'MSE', 'R^2']
        list_of_metrics_types = ['Treinamento', 'Validação' ]
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
                    df_filter = metrics_df['Municipio Treinado'] == model_name
                    average = statistics.mean( metrics_df[ df_filter ][f'{metric_name} {metric_type}'].to_list() )
                    metrics_averages_dict[metric_name][metric_type][model_name] = average
        
        # Plotting the graphs:
        for metric_name in list_of_metrics_names:
            Y_axis = np.arange(len(list_of_models_names)) 
            
            # 0.4: width of the bars; 0.2: distance between the groups
            plt.barh(Y_axis - 0.2, metrics_averages_dict[metric_name]['Treinamento'].values(), 0.4, label = 'Training'  )
            plt.barh(Y_axis + 0.2, metrics_averages_dict[metric_name]['Validação'  ].values(), 0.4, label = 'Validation')
            
            plt.yticks(Y_axis, list_of_models_names, rotation=45)
            plt.ylabel("Machine Learning models")
            plt.xlabel(f'Average {metric_name}' if metric_name != 'R^2' else 'Average R²')
            plt.title ("Comparison of performance of different models")
            plt.legend()
            
            self._saveFig(plt, f'Bar Plots. {metric_name}.')
            plt.close()
    
    def define_normal_distribution(self, axis, x_values):
        if np.std(x_values) > 0:
            mu  , std  = norm.fit     (x_values)
            xmin, xmax = axis.get_xlim()
            x          = np.linspace  (xmin, xmax, 100)
            p          = norm.pdf     (x   , mu  , std)
            
            return x, p
        else:
            print('Info: normal distribution <= 0')
            return 0, 0
    
    def drawMetricsHistograms(self, metrics_df):
        COLS_LABELS = ['Training (columns)', 'Validation (columns)']
        COLS_COLORS = [        'red'       ,        'green'        ]
        
        # Creation of the empty dictionary:
        list_of_metrics_names = [    'MAE'    ,   'RMSE'   ]
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
                    df_filter = metrics_df['Municipio Treinado'] == model_name
                    metrics_dict[metric_name][metric_type][model_name] = metrics_df[ df_filter ][f'{metric_name} {metric_type}'].to_list()
    
        # Plotting the graphs:
        for model_name in list_of_models_names:
            x_MAE  = [ metrics_dict['MAE' ]['Treinamento'][model_name] ,
                       metrics_dict['MAE' ]['Validação'  ][model_name] ]
            x_RMSE = [ metrics_dict['RMSE']['Treinamento'][model_name] ,
                       metrics_dict['RMSE']['Validação'  ][model_name] ]
            
            # fig = plt.figure(figsize=(12, 8))
            fig = plt.figure()
            gs  = gridspec.GridSpec(2, 2, height_ratios=[10, 1])

            # LAYOUT:
            # +------------------------+------------------------+
            # |           MAE          |          RMSE          |
            # +------------------------+------------------------+
            # |                      LEGEND                     |
            # +-------------------------------------------------+

            ax_mae    = fig.add_subplot(gs[0, 0])
            ax_rmse   = fig.add_subplot(gs[0, 1])
            ax_legend = fig.add_subplot(gs[1, :])
            
            # MAE Histogram
            ax_mae.hist(x_MAE , density=True, histtype='bar', color=COLS_COLORS, label=COLS_LABELS)
            x, p = self.define_normal_distribution(ax_mae, x_MAE[0])
            ax_mae.plot(x, p, 'red', linewidth=2, label='Training Normal Distribution (curves)')
            x, p = self.define_normal_distribution(ax_mae, x_MAE[1])
            ax_mae.plot(x, p, 'green', linewidth=2, label='Validation Normal Distribution (curves)')
            ax_mae.set_title('MAE')
            ax_mae.set_ylabel('Frequency')
            
            # RMSE Histogram
            ax_rmse.hist(x_RMSE, density=True, histtype='bar', color=COLS_COLORS, label=COLS_LABELS)
            x, p = self.define_normal_distribution(ax_rmse, x_RMSE[0])
            ax_rmse.plot(x, p, 'red', linewidth=2, label='Training Normal Distribution (curves)')
            x, p = self.define_normal_distribution(ax_rmse, x_RMSE[1])
            ax_rmse.plot(x, p, 'green', linewidth=2, label='Validation Normal Distribution (curves)')
            ax_rmse.set_title('RMSE')
            
            # Plot legend in separate subplot
            ax_legend.axis('off')
            handles, labels = ax_mae .get_legend_handles_labels()
            ax_legend.legend(
                handles, labels,
                loc='center', ncol=2, frameon=False
            )
            
            fig.suptitle(f'Histograms of model {model_name}')
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            
            self._saveFig(plt, 'Histograms.', model_name, model_name)
            plt.close()
    
    def showResidualPlots(self, true_values_dict, predicted_values_dict, city_cluster_name, city_for_training, city_for_predicting):
        residuals        = {'Train': true_values_dict['Train'] - predicted_values_dict['Train'],
                            'Test' : true_values_dict['Test' ] - predicted_values_dict['Test' ]}
        
        for training_or_testing in ['Train', 'Test']:
            plt.scatter(predicted_values_dict[training_or_testing], residuals[training_or_testing], alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title (f'Residual Plot for {training_or_testing} data.\nModel {city_for_training} applied to {city_for_predicting}.')
            
            self._saveFig(plt, f'Residual Plots {training_or_testing}', city_cluster_name, city_for_training, city_for_predicting)
            plt.close()
    
    def showR2ScatterPlots(self, true_values_dict, predicted_values_dict, city_cluster_name, city_for_training, city_for_predicting):    
        for training_or_testing in ['Train', 'Test']:
            plt.scatter(true_values_dict[training_or_testing], predicted_values_dict[training_or_testing], label = 'R²')
            
            # Generates a single line by creating `x_vals`, a sequence of 100 evenly spaced values between the min and max values in true_values
            flattened_values = np.ravel(true_values_dict[training_or_testing])
            x_vals = np.linspace(min(flattened_values), max(flattened_values), 100)
            plt.plot(x_vals, x_vals, color='red', label='x=y')  # Line will only appear once
            
            plt.title (f'R² {training_or_testing} data. Model {city_for_training} applied to {city_for_predicting}.')
            plt.xlabel('True values')
            plt.ylabel('Predicted values')
            plt.legend()
                
            self._saveFig(plt, f'R² Scatter Plot {training_or_testing}', city_cluster_name, city_for_training, city_for_predicting)
            plt.close()
    
    def drawMetricsRadarPlots(self, metrics_df):
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
                    df_filter = metrics_df['Municipio Treinado'] == model_name
                    average = statistics.mean( metrics_df[ df_filter ][f'{metric_name} {metric_type}'].to_list() )
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
                
                self._saveFig(plt, f'Radar Plots. {model_name}. {metric_name}. {metric_type}.', model_name, model_name)
                plt.close()
    
    def showTaylorDiagrams(self, metrics_df, city_cluster_name, city_for_training, city_for_predicting):
        
        label =          ['Obs', 'Train', 'Test']
        sdev  = np.array([metrics_df.iloc[-1]['Desvio Padrão Obs.'             ] ,
                          metrics_df.iloc[-1]['Desvio Padrão Pred. Treinamento'] ,
                          metrics_df.iloc[-1]['Desvio Padrão Pred. Validação'  ] ])
        ccoef = np.array([1.                                                     ,
                          metrics_df.iloc[-1]['Coef. de Correlação Treinamento'] ,
                          metrics_df.iloc[-1]['Coef. de Correlação Validação'  ] ])
        rmse  = np.array([0.                                                     ,
                          metrics_df.iloc[-1]['RMSE Treinamento'               ] ,
                          metrics_df.iloc[-1]['RMSE Validação'                 ] ])
        
        # Plotting:
        ## If both are positive, 90° (2 squares), if one of them is negative, 180° (2 rectangles)
        figsize = (2*8, 2*5) if (metrics_df.iloc[-1]['Coef. de Correlação Treinamento'] > 0 and metrics_df.iloc[-1]['Coef. de Correlação Validação'] > 0) else (2*8, 2*3)
        
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)
        AVAILABLE_AXES = {'a) Training': 0, 'b) Testing': 1}
        for axs_title, axs_number in AVAILABLE_AXES.items():
            ax = axs[axs_number]
            ax.set_title(axs_title, loc="left", y=1.1)
            ax.set(adjustable='box', aspect='equal')
            sm.taylor_diagram(ax, sdev, rmse, ccoef, markerLabel = label, markerLabelColor = 'r', 
                              markerLegend = 'on', markerColor   = 'r' ,
                              styleOBS     = '-' , colOBS        = 'r' ,       markerobs = 'o',
                              markerSize   =   6 , tickRMS       = [0.0, 0.05, 0.1, 0.15, 0.2],
                              tickRMSangle = 115 , showlabelsRMS = 'on',
                              titleRMS     = 'on', titleOBS      = 'Obs')
        plt.suptitle (f'Model {city_for_training} applied to {city_for_predicting}')
        fig.tight_layout(pad = 1.5)
        
        self._saveFig(plt, 'Taylor Diagram.', city_cluster_name, city_for_training, city_for_predicting)
        plt.close()