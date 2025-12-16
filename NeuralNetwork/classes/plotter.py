import os
import matplotlib.pyplot   as      plt
import numpy               as       np


class Plotter:
    
    OUTPUT_DIR_ADDR   = './Output/'
    
    METRICS_PORTIONS_CENTRAL   = [ '80%', '20%']
    METRICS_PORTIONS_BORDERING = ['100%', '20%']
    
    def _saveFig(self, plot, filename, city_cluster_name=None, city_for_training=None, city_for_predicting=None):
        if city_for_predicting:
            FILEPATH = f'./{Plotter.OUTPUT_DIR_ADDR}/cluster {city_cluster_name}/model {city_for_training}/city {city_for_predicting}/'
            os.makedirs(FILEPATH, exist_ok=True)
            plt.savefig(FILEPATH + filename + f' - Model {city_for_training} applied to {city_for_predicting}.png')
        elif city_for_training:
            FILEPATH = f'./{Plotter.OUTPUT_DIR_ADDR}/cluster {city_cluster_name}/model {city_for_training}/'
            os.makedirs(FILEPATH, exist_ok=True)
            plt.savefig(FILEPATH + filename + f' - Model {city_for_training}.png')
        else:
            FILEPATH = './{Plotter.OUTPUT_DIR_ADDR}/'
            os.makedirs(FILEPATH, exist_ok=True)
            plt.savefig(FILEPATH + filename, bbox_inches="tight")

    def plotDatasetPlots(self, dataset, spei_test, split, city_cluster_name, city_for_training, city_for_predicting):
        self.showSpeiData(dataset     , spei_test, split, city_cluster_name, city_for_training, city_for_predicting)
        self.showSpeiTest(dataset     , spei_test, split, city_cluster_name, city_for_training, city_for_predicting)

    def plotModelPlots(self                  , dataset, spei_dict            , is_model           ,
                       spei_expected_outputs , spei_predicted_values,
                       monthForPredicted_dict, has_trained          ,
                       history               , metrics_df           ,
                       city_cluster_name     , city_for_training    , city_for_predicting):
        
        self.showResidualPlots           (is_model         , spei_expected_outputs, spei_predicted_values,
                                          city_cluster_name, city_for_training    , city_for_predicting  )
        self.showR2ScatterPlots          (is_model         , spei_expected_outputs, spei_predicted_values,
                                          city_cluster_name, city_for_training    , city_for_predicting  )
        self.showPredictionsDistribution (dataset, is_model         , spei_expected_outputs, spei_predicted_values,
                                          city_cluster_name, city_for_training    , city_for_predicting  )
        self.showPredictionResults       (dataset, is_model         , spei_expected_outputs, spei_predicted_values , monthForPredicted_dict,
                                          city_cluster_name, city_for_training   , city_for_predicting)
    
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
        
        y1positive = np.array(speiValues)>=0
        y1negative = np.array(speiValues)<=0
    
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
    
    def _calculateDenormalizedValues(self, dataset, is_model, spei_expected_outputs, spei_predicted_values):
        speiValues           = dataset.get_spei           ()
        
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
        spei_max_value = np.max(speiValues)
        spei_min_value = np.min(speiValues)
        
        spei_delta     = spei_max_value - spei_min_value
        ###CALCULATIONS########################################################
        true_values_denormalized_dict['100%'] = (spei_expected_outputs ['100%']           * spei_delta + spei_min_value)
        true_values_denormalized_dict[ '20%'] = (spei_expected_outputs [ '20%']           * spei_delta + spei_min_value)
        
        predictions_denormalized_dict['100%'] = (spei_predicted_values['100%']           * spei_delta + spei_min_value)
        predictions_denormalized_dict[ '20%'] = (spei_predicted_values[ '20%'].flatten() * spei_delta + spei_min_value)
        
        return true_values_denormalized_dict, predictions_denormalized_dict
    
    def showPredictionResults(self      ,    dataset, is_model   , spei_expected_outputs, spei_predicted_values,
                              months_for_expected_outputs, city_cluster_name   , city_for_training    , city_for_predicting):
        
        (trueValues_denormalized ,
         predictions_denormalized) = self._calculateDenormalizedValues(dataset, is_model, spei_expected_outputs, spei_predicted_values)
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
    
    def showPredictionsDistribution(self, dataset, is_model   , spei_expected_outputs, spei_predicted_values,
                                    city_cluster_name, city_for_training   , city_for_predicting  ):
        
        (trueValues_denormalized ,
         predictions_denormalized) = self._calculateDenormalizedValues(dataset, is_model, spei_expected_outputs, spei_predicted_values)
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