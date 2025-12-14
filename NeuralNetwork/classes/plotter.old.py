# import statistics                      # Unused for now
# import skill_metrics       as       sm # Unused for now
# import matplotlib.gridspec as gridspec # Unused for now
# from   scipy.stats         import norm # Unused for now

class PlotterDisabled:
    
    # Disabled, as these are not going to be used on Anderson's masters dissertation:
    # def plotMetricsPlots(self, metrics_df):
    #     self.drawMetricsBoxPlots   (metrics_df)
    #     self.drawMetricsBarPlots   (metrics_df)
    #     self.drawMetricsHistograms (metrics_df)
        
    #     # Issue #3: "Radar Plots are an unfinished work"
    #     # https://github.com/A-Infor/Python-OOP-LSTM-Drought-Predictor/issues/3
    #     # self.drawMetricsRadarPlots (metrics_df)
    
    def plotModelPlots(self                  , spei_dict            , is_model           ,
                       spei_expected_outputs , spei_predicted_values,
                       monthForPredicted_dict, has_trained          ,
                       history               , metrics_df           ,
                       city_cluster_name     , city_for_training    , city_for_predicting):

        # Issue #7: "Taylor Diagrams are an unfinished work":
        # https://github.com/A-Infor/Python-OOP-LSTM-Drought-Predictor/issues/7
        # self.showTaylorDiagrams         (metrics_df                             , city_cluster_name, city_for_training, city_for_predicting)
        pass
        
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