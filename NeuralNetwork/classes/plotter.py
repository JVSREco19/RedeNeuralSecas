import os
import matplotlib.pyplot   as      plt
import numpy               as       np


class Plotter:
    
    OUTPUT_DIR_ADDR   = './Output/'
    
    METRICS_PORTIONS_CENTRAL   = [   '80%'  ,   '20%'  ]
    METRICS_PORTIONS_BORDERING = [              '20%'  ]
    METRICS_TECHNIQUES         = ['tumbling', 'sliding']
    
    def _saveFig(self, plot, file_title, city_cluster_name=None, city_for_training=None, city_for_predicting=None, technique=None):
        
        if city_for_predicting:
            FILEPATH = f'./{Plotter.OUTPUT_DIR_ADDR}/cluster {city_cluster_name}/model {city_for_training}/city {city_for_predicting}/'
            
            if technique != None:
                FILENAME = file_title + f' - Model {city_for_training} applied to {city_for_predicting} - {technique}.png'
            else:
                FILENAME = file_title + f' - Model {city_for_training} applied to {city_for_predicting}.png'
                
        elif city_for_training:
            FILEPATH = f'./{Plotter.OUTPUT_DIR_ADDR}/cluster {city_cluster_name}/model {city_for_training}/'    

            if technique != None:
                FILENAME = file_title + f' - Model {city_for_training} - {technique}.png'
            else:
                FILENAME = file_title + f' - Model {city_for_training}.png'
           
        else:
            FILEPATH = './{Plotter.OUTPUT_DIR_ADDR}/'
            FILENAME = file_title
            # plt.savefig(FILEPATH + file_title, bbox_inches="tight")
        
        os.makedirs(FILEPATH, exist_ok=True)
        plt.savefig(FILEPATH + FILENAME)

    def plotDatasetPlots(self, dataset, spei_test, split, city_cluster_name, city_for_training, city_for_predicting):
        self.showSpeiData(dataset     , spei_test, split, city_cluster_name, city_for_training, city_for_predicting)
        self.showSpeiTest(dataset     , spei_test, split, city_cluster_name, city_for_training, city_for_predicting)

    def plotModelPlots(self,     dataset,       spei_dict,        is_model,
        spei_data     ,   months_data  , has_trained                      ,
        spei_predicted_values_tumbling, spei_predicted_values_sliding     ,
        history                                                           ,
        city_cluster_name,     city_for_training,      city_for_predicting):

        spei_predicted_values = {'tumbling': spei_predicted_values_tumbling,
                                 'sliding': spei_predicted_values_sliding  }

        # Un-windowed months, one entry per month, mirroring the
        # train_test_split that produced spei_dict. `spei_dict['100%']` is
        # the full un-windowed series, so its length matches the full
        # month axis available via dataset.get_months().
        split_position = len(spei_dict['80%'])
        full_months = dataset.get_months()
        months_dict = {
            '80%' : full_months[:split_position],
            '20%' : full_months[ split_position:],
            '100%': full_months
        }

        for technique in Plotter.METRICS_TECHNIQUES:
            # self.showResidualPlots           (is_model         , spei_expected_outputs, spei_predicted_values,
                                              # city_cluster_name, city_for_training    , city_for_predicting  , technique)
            # self.showR2ScatterPlots          (is_model         , spei_expected_outputs, spei_predicted_values,
                                              # city_cluster_name, city_for_training    , city_for_predicting  , technique)
            # self.showPredictionsDistribution (dataset, is_model         , spei_expected_outputs, spei_predicted_values,
                                              # city_cluster_name, city_for_training    , city_for_predicting  , technique)

            self.showPredictionResults       (dataset, spei_dict, months_dict, is_model, spei_data, spei_predicted_values, months_data,
                                              city_cluster_name, city_for_training   , city_for_predicting   , technique)
    
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
    
    def _calculateDenormalizedValues(self, dataset, is_model, spei_expected_outputs, spei_predicted_values, technique):
   
        ###ADJUSTMENTS OF INPUTS###############################################
        spei_expected_outputs    [ '20%'] = spei_expected_outputs[ '20%'].flatten()
        
        if is_model:
            spei_expected_outputs['100%'] = spei_expected_outputs['100%'].flatten()
            
            spei_predicted_values[technique]['100%'] = np.append(
                spei_predicted_values[technique]['80%'],
                spei_predicted_values[technique]['20%'])
        
        ###PREPARATIVES FOR OUTPUT#############################################
        if is_model:
            RELEVANT_PORTIONS = ['100%', '20%']
        else:
            RELEVANT_PORTIONS = [        '20%']
        
        true_values_denormalized_dict = dict.fromkeys(RELEVANT_PORTIONS)
        predictions_denormalized_dict = dict.fromkeys(RELEVANT_PORTIONS)
        
        ###MIN & MAX FOR CALCULATION###########################################
        # Use normalization parameters from training set only
        spei_max_value = dataset.spei_max
        spei_min_value = dataset.spei_min
        
        spei_delta     = spei_max_value - spei_min_value
        ###CALCULATIONS########################################################
        # Handle zero variance case
        if np.isclose(spei_delta, 0):
            # If delta is 0, denormalized values should be constant at spei_min_value
            if is_model:
                true_values_denormalized_dict['100%'] = np.full_like(spei_expected_outputs['100%'], spei_min_value)
                predictions_denormalized_dict['100%'] = np.full_like(spei_predicted_values['100%'], spei_min_value)
            
            true_values_denormalized_dict[ '20%'] = np.full_like(spei_expected_outputs[ '20%'], spei_min_value)
            flattened_20 = spei_predicted_values['20%'].flatten()
            predictions_denormalized_dict[ '20%'] = np.full_like(flattened_20, spei_min_value)
        else:
            if is_model:
                true_values_denormalized_dict['100%'] = (spei_expected_outputs           ['100%']           * spei_delta + spei_min_value)
                predictions_denormalized_dict['100%'] = (spei_predicted_values[technique]['100%']           * spei_delta + spei_min_value)
            
            true_values_denormalized_dict[ '20%']     = (spei_expected_outputs           [ '20%']           * spei_delta + spei_min_value)
            predictions_denormalized_dict[ '20%']     = (spei_predicted_values[technique][ '20%'].flatten() * spei_delta + spei_min_value)
        
        if is_model:
            assert '100%' in true_values_denormalized_dict, f'There is no 100% portion for true_values_denormalized_dict of city {dataset.city_name} from cluster {dataset.city_cluster_name} using technique {technique}'
            assert '100%' in predictions_denormalized_dict, f'There is no 100% portion for predictions_denormalized_dict of city {dataset.city_name} from cluster {dataset.city_cluster_name} using technique {technique}'
            assert true_values_denormalized_dict['100%'].shape == predictions_denormalized_dict['100%'].shape,\
            f"{true_values_denormalized_dict['100%'].shape} != {predictions_denormalized_dict['100%'].shape}"
            
        assert  '20%' in true_values_denormalized_dict, f'There is no  20% portion for true_values_denormalized_dict of city {dataset.city_name} from cluster {dataset.city_cluster_name} using technique {technique}'
        assert  '20%' in predictions_denormalized_dict, f'There is no  20% portion for predictions_denormalized_dict of city {dataset.city_name} from cluster {dataset.city_cluster_name} using technique {technique}'
        assert true_values_denormalized_dict['20%'].shape == predictions_denormalized_dict['20%'].shape,\
        f"{true_values_denormalized_dict['20%'].shape} != {predictions_denormalized_dict['20%'].shape}"
        
        # print(f'OK: city {dataset.city_name} from cluster {dataset.city_cluster_name} using technique {technique}!')
        
        return true_values_denormalized_dict, predictions_denormalized_dict
    
    def _aggregate_predictions_by_month(self, values_2d, months_2d):
        """
        Group overlapping sliding-window values by the month they predict, then
        average per month. Returns (unique_months_1d, averaged_values_1d) with
        identical length, sorted chronologically.
        """
        flat_months = np.asarray(months_2d).flatten()
        flat_values = np.asarray(values_2d).flatten()

        unique_months   = np.unique(flat_months)            # sorted, dtype preserved
        averaged_values = np.array([
            flat_values[flat_months == m].mean() for m in unique_months
        ])

        return unique_months, averaged_values

    def _spans_by_kind(self, plot_months, predicted_values):
        """
        Find the contiguous ranges of `plot_months` over which the predicted
        series is NaN, and classify each run by kind:

          - 'lookback': a NaN run that the model genuinely has no prediction
            for, because the first window(s) of some slice consume those
            months as their lookback portion. These appear at the start of a
            slice (`start == 0`) and in the middle of a concatenated series
            where one slice's lookback sits between two predicted regions.
          - 'unused': a NaN run at the END of the series that no window
            covers, because the data length isn't a multiple of the window
            step. These are an arithmetic leftover, not a real lookback.

        Each span is positioned at the midpoint between the last predicted
        month before the run and the first predicted month after it (or
        extrapolated half a step past the series if the run touches an edge),
        so the band ends exactly at the visual boundary between the last
        NaN month and the next predicted month, regardless of whether the
        x-axis is numerical or datetime.

        Datetime axes are handled by converting to integer representation
        (months-since-epoch for `datetime64[M]`, day count for finer
        resolutions) for the midpoint arithmetic, then converting back.

        Returns a list of (kind, x_left, x_right) tuples.
        """
        plot_months      = np.asarray(plot_months)
        predicted_values = np.asarray(predicted_values)

        is_datetime = np.issubdtype(plot_months.dtype, np.datetime64)
        if is_datetime:
            # numpy.datetime64 arithmetic on the raw array works in the
            # array's own resolution. We promote to nanosecond resolution
            # so that a half-step on any axis (15 days for monthly, 12
            # hours for daily, etc.) is representable exactly instead of
            # being rounded to the nearest unit. matplotlib renders
            # nanosecond-precision x-values at the correct sub-tick
            # position on coarser axes.
            as_int = plot_months.astype('datetime64[ns]').view(np.int64)
            back   = lambda v: np.datetime64(int(round(v)), 'ns')
        else:
            # Numerical axes (int, float, etc.): keep the values as float64
            # so that the band boundary falls on the visual half-step
            # between two ticks (e.g. 5.5 between ticks 5 and 6), instead
            # of snapping to an integer tick and overlapping the predicted
            # point. Using `astype(np.int64)` here would lose sub-integer
            # information on a half-step axis.
            as_int = plot_months.astype(np.float64)
            back   = lambda v: float(v)

        isnan = np.isnan(predicted_values)
        spans = []
        n = plot_months.shape[0]
        i = 0
        while i < n:
            if not isnan[i]:
                i += 1
                continue
            start = i
            while i < n and isnan[i]:
                i += 1
            # i is now the index of the FIRST non-NaN month after the run
            # (or n, if the run extends to the end of the series).
            last_nan_idx = i - 1   # index of the last NaN month in the run

            # A NaN run that reaches the end of the series with no predicted
            # month after it is an arithmetic leftover from the sliding
            # window step, not a real lookback. Any run that has a predicted
            # month on both sides (or at least on the right side, i.e. a
            # leading run that starts at index 0) is a genuine lookback.
            if i >= n and start > 0:
                kind = 'unused'
            else:
                kind = 'lookback'

            # x_left: midpoint between the month just before the run
            # and the first NaN month. If the run starts at index 0,
            # there is no month before it; extrapolate half a step backwards.
            if start == 0:
                if n >= 2:
                    step = as_int[1] - as_int[0]
                else:
                    step = np.float64(1)   # degenerate single-element case
                x_left_int = as_int[0] - step / 2
            else:
                x_left_int = (as_int[start - 1] + as_int[start]) / 2

            # x_right: midpoint between the last NaN month and the first
            # predicted month after the run. If the run reaches the end of
            # the series, extrapolate half a step forwards.
            if i >= n:
                if n >= 2:
                    step = as_int[-1] - as_int[-2]
                else:
                    step = np.float64(1)
                x_right_int = as_int[-1] + step / 2
            else:
                x_right_int = (as_int[last_nan_idx] + as_int[i]) / 2

            spans.append((kind, back(x_left_int), back(x_right_int)))
        return spans

    def _draw_no_prediction_bands(self, plot_months, predicted_values):
        """
        Draw a translucent vertical band over each region of the current
        axes where the predicted series is NaN. Two kinds of regions are
        distinguished:

          - 'Lookback' (dark, alpha 0.20): the model genuinely has no
            prediction for those months because they're consumed as the
            lookback portion of the first window(s) of some slice.
          - 'Unused'  (green, alpha 0.20): months at the end of the series
            that no window covers because the data length isn't a multiple
            of the window step. These are an arithmetic leftover.

        Returns a list of patch handles (one per band); each kind shares
        a single legend label, so the caller should de-duplicate the
        legend entries by label.
        """
        style = {
            'lookback': dict(color='black', alpha=0.20, label='Lookback'),
            'unused'  : dict(color='green', alpha=0.20, label='Unused'  ),
        }
        handles = []
        for kind, x_left, x_right in self._spans_by_kind(plot_months, predicted_values):
            patch = plt.axvspan(x_left, x_right, **style[kind])
            handles.append(patch)
        return handles

    @staticmethod
    def _dedupe_legend_handles(handles):
        """
        Drop duplicate legend entries that share the same label. Keeps the
        first handle seen per label; entries with no label (or the
        '_nolegend_' sentinel) are kept as-is.
        """
        seen = set()
        deduped = []
        for h in handles:
            label = getattr(h, '_label', None)
            if label is None or label == '_nolegend_':
                deduped.append(h)
                continue
            if label in seen:
                continue
            seen.add(label)
            deduped.append(h)
        return deduped

    def showPredictionResults(self, dataset, spei_dict, months_dict, is_model, spei_data, spei_predicted_values, months_data,
                              city_cluster_name, city_for_training, city_for_predicting, technique):

        (trueValues_denormalized ,
         predictions_denormalized) = self._calculateDenormalizedValues(dataset, is_model,
                                          spei_data[technique]['output'], spei_predicted_values, technique)

        # The un-windowed real series (denormalized), one entry per month.
        # For tumbling, this is the real line we want to plot so every real
        # month appears, including the lookback months that have no predicted
        # counterpart. The predicted line stays aligned to its windowed months;
        # lookback months stay NaN on the predicted axis and matplotlib will
        # leave a gap there instead of drawing a diagonal "bridge".
        spei_delta = dataset.spei_max - dataset.spei_min
        if np.isclose(spei_delta, 0):
            full_real_20  = np.full_like(spei_dict[ '20%'], dataset.spei_min)
            full_real_100 = np.full_like(spei_dict['100%'], dataset.spei_min)
        else:
            full_real_20  = spei_dict[ '20%'] * spei_delta + dataset.spei_min
            full_real_100 = spei_dict['100%'] * spei_delta + dataset.spei_min
        # The un-windowed months: one entry per month, in chronological order,
        # matching the real series above. months_dict is built in plotModelPlots
        # from dataset.get_months() and the train_test_split boundary.
        months_axis_20  = months_dict[ '20%']
        months_axis_100 = months_dict['100%']

        ###100%################################################################
        if is_model:
            if technique == 'sliding':
                # Real line uses the FULL un-windowed series so every real
                # month is plotted, including the lookback months that have
                # no predicted counterpart. The predicted line is the
                # per-month average of the overlapping window outputs,
                # placed at the corresponding real months; lookback months
                # stay NaN on the predicted axis and matplotlib leaves a gap.
                plot_months_100         = months_axis_100
                trueValues_to_plot_100  = full_real_100
                predictions_to_plot_100 = np.full(months_axis_100.shape[0], np.nan)
                months_100 = months_data[technique]['output']['100%']
                _avg_months_100, avg_pred_100 = self._aggregate_predictions_by_month(
                    predictions_denormalized['100%'], months_100)
                for m, p in zip(_avg_months_100, avg_pred_100):
                    idx = np.searchsorted(plot_months_100, m)
                    if idx < plot_months_100.shape[0] and plot_months_100[idx] == m:
                        predictions_to_plot_100[idx] = p
            else:
                # Tumbling: real line uses the full un-windowed series, so
                # every real month is plotted. The predicted line is aligned
                # to the months its windows actually cover; lookback months
                # stay NaN on the predicted axis.
                plot_months_100       = months_axis_100
                trueValues_to_plot_100 = full_real_100
                predictions_to_plot_100 = np.full(months_axis_100.shape[0], np.nan)
                pred_months_100 = months_data[technique]['output']['100%'].flatten()
                pred_values_100 = predictions_denormalized['100%']
                for m, p in zip(pred_months_100, pred_values_100):
                    idx = np.searchsorted(plot_months_100, m)
                    if idx < plot_months_100.shape[0] and plot_months_100[idx] == m:
                        predictions_to_plot_100[idx] = p

            plt.figure ()

            assert plot_months_100.shape[0] == trueValues_to_plot_100.shape[0] == predictions_to_plot_100.shape[0],\
            f"{plot_months_100.shape} != {trueValues_to_plot_100.shape} != {predictions_to_plot_100.shape}"

            real_line_100     , = plt.plot(plot_months_100, trueValues_to_plot_100 , label='Real'      )
            predicted_line_100, = plt.plot(plot_months_100, predictions_to_plot_100, label='Predicted' )

            # Translucent bands over each no-prediction region: dark for
            # genuine lookbacks, green for unused trailing leftovers.
            no_pred_handles_100 = self._draw_no_prediction_bands(plot_months_100, predictions_to_plot_100)

            split_handle_100 = plt.axvline(months_dict['80%'][-1], color='r',
                                            label='Test portion start')

            plt.legend(handles=Plotter._dedupe_legend_handles(
                          [real_line_100, predicted_line_100, *no_pred_handles_100, split_handle_100]),
                       loc='best')
            plt.xlabel ('Year')
            plt.ylabel ('SPEI')
            plt.title  (f'Model {city_for_training} applied to {city_for_predicting}:\nreal and predicted SPEI values (100%\'s {technique})')
            # plt.show()

            self._saveFig(plt, 'Previsao 100%', city_cluster_name, city_for_training, city_for_predicting, technique)
            plt.close()
        ###20%#################################################################
        if technique == 'sliding':
            # Real line uses the FULL un-windowed 20% series, so every real
            # month is plotted. The predicted line is the per-month average
            # of the overlapping window outputs, placed at the corresponding
            # real months; lookback months stay NaN on the predicted axis.
            plot_months_20         = months_axis_20
            trueValues_to_plot_20  = full_real_20
            predictions_to_plot_20 = np.full(months_axis_20.shape[0], np.nan)
            months_20 = months_data[technique]['output']['20%']
            _avg_months_20, avg_pred_20 = self._aggregate_predictions_by_month(
                predictions_denormalized['20%'], months_20)
            for m, p in zip(_avg_months_20, avg_pred_20):
                idx = np.searchsorted(plot_months_20, m)
                if idx < plot_months_20.shape[0] and plot_months_20[idx] == m:
                    predictions_to_plot_20[idx] = p
        else:
            # Tumbling: real line uses the full un-windowed 20% series, so
            # every real month is plotted. The predicted line is aligned to
            # the months its windows actually cover; lookback months stay
            # NaN on the predicted axis and matplotlib leaves a gap.
            plot_months_20       = months_axis_20
            trueValues_to_plot_20 = full_real_20
            predictions_to_plot_20 = np.full(months_axis_20.shape[0], np.nan)
            pred_months_20 = months_data[technique]['output']['20%'].flatten()
            for m, p in zip(pred_months_20, predictions_denormalized['20%']):
                idx = np.searchsorted(plot_months_20, m)
                if idx < plot_months_20.shape[0] and plot_months_20[idx] == m:
                    predictions_to_plot_20[idx] = p

        plt.figure ()

        assert plot_months_20.shape[0] == trueValues_to_plot_20.shape[0] == predictions_to_plot_20.shape[0],\
        f"{plot_months_20.shape} != {trueValues_to_plot_20.shape} != {predictions_to_plot_20.shape}"

        real_line_20     , = plt.plot(plot_months_20, trueValues_to_plot_20 , label='Real'      )
        predicted_line_20, = plt.plot(plot_months_20, predictions_to_plot_20, label='Predicted' )

        # Translucent bands over each no-prediction region: dark for
        # genuine lookbacks, green for unused trailing leftovers.
        no_pred_handles_20 = self._draw_no_prediction_bands(plot_months_20, predictions_to_plot_20)

        plt.legend(handles=Plotter._dedupe_legend_handles(
                      [real_line_20, predicted_line_20, *no_pred_handles_20]),
                   loc='best')
        plt.xlabel ('Year')
        plt.ylabel ('SPEI')
        plt.title  (f'Model {city_for_training} applied to {city_for_predicting}:\nreal and predicted SPEI values (20%\'s {technique})')
        # plt.show()

        self._saveFig(plt, 'Previsao 20%', city_cluster_name, city_for_training, city_for_predicting, technique)
        plt.close()
        #######################################################################
    
    def showPredictionsDistribution(self, dataset, is_model   , spei_expected_outputs, spei_predicted_values,
                                    city_cluster_name, city_for_training   , city_for_predicting, technique  ):
        
        (trueValues_denormalized ,
         predictions_denormalized) = self._calculateDenormalizedValues(dataset, is_model, spei_expected_outputs, spei_predicted_values)
        ###100%################################################################
        if is_model:
            plt.figure ()
            plt.scatter(x =  trueValues_denormalized['100%'],
                        y = predictions_denormalized['100%'],
                        color=['white'],  marker='^', edgecolors='black')
            plt.xlabel ('Real SPEI')
            plt.ylabel ('Predicted SPEI'  )
            plt.axline ( (0, 0) , slope=1 )
            plt.title  (f'Model {city_for_training} applied to {city_for_predicting}:\nSPEI (100%\'s distribution {technique})')
            #plt.show()
            
            self._saveFig(plt, 'distribuiçãoDoSPEI 100%', city_cluster_name, city_for_training, city_for_predicting, technique)
            plt.close()
        ###20%#################################################################
        plt.figure ()
        plt.scatter(x =  trueValues_denormalized[ '20%'],
                    y = predictions_denormalized[ '20%'],
                    color=['white'],  marker='D', edgecolors='black')
        plt.xlabel ('Real SPEI')
        plt.ylabel ('Predicted SPEI'  )
        plt.axline ( (0, 0) , slope=1 )
        plt.title  (f'Model {city_for_training} applied to {city_for_predicting}:\nSPEI (20%\'s distribution {technique})')
        #plt.show()
        
        self._saveFig(plt, 'distribuiçãoDoSPEI 20%', city_cluster_name, city_for_training, city_for_predicting, technique)
        plt.close()
        #######################################################################

    def drawModelLineGraph(self, history, technique, city_cluster_name, city_for_training):
        y_mae = history.history['mae'][19:]
        y_rmse = history.history['rmse'][19:]
        y_mse = history.history['mse'][19:]
        y_r2 = history.history['r2'][19:]
    
        x = range(20, 20 + len(y_mae))
    
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    
        axs[0, 0].plot(x, y_mae, 'tab:blue')
        axs[0, 0].set_title('MAE')
    
        axs[0, 1].plot(x, y_rmse, 'tab:orange')
        axs[0, 1].set_title('RMSE')
    
        axs[1, 0].plot(x, y_mse, 'tab:green')
        axs[1, 0].set_title('MSE')
    
        axs[1, 1].plot(x, y_r2, 'tab:red')
        axs[1, 1].set_title('R²')
    
        ticks = [20, 50, 100, 150]
        ticks = [t for t in ticks if t <= x[-1]]
    
        for ax in axs.flat:
            ax.set_xticks(ticks)
            ax.set_xlim(20, x[-1])
    
        for ax in axs[1]:
            ax.set(xlabel='Epochs (training)')
    
        plt.suptitle(f'Model {city_for_training} ({technique})')
        self._saveFig(plt, 'Line Graph.', city_cluster_name=city_cluster_name, city_for_training=city_for_training, technique=technique)
        plt.close()
    
        def define_box_properties(self, plot_name, color_code, label):
            	for k, v in plot_name.items():
            		plt.setp(plot_name.get(k), color=color_code)
            		
            	# use plot function to draw a small line to name the legend.
            	plt.plot([], c=color_code, label=label)
            	plt.legend()
    
    def showResidualPlots(self  ,  is_model, true_values_dict , predicted_values_dict,
                          city_cluster_name, city_for_training, city_for_predicting, technique  ):
        
        if is_model:
            residuals        = { '80%': true_values_dict[ '80%'] - predicted_values_dict[ '80%'],
                                 '20%': true_values_dict[ '20%'] - predicted_values_dict[ '20%']}
        else:
            residuals        = { '20%': true_values_dict[ '20%'] - predicted_values_dict[ '20%']}
        
        for data_portion_type in Plotter.METRICS_PORTIONS_CENTRAL if is_model else Plotter.METRICS_PORTIONS_BORDERING:
            plt.scatter(predicted_values_dict[data_portion_type], residuals[data_portion_type], alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title (f'Residual Plot for {data_portion_type} data ({technique}).\nModel {city_for_training} applied to {city_for_predicting}.')
            
            self._saveFig(plt, f'Residual Plots {data_portion_type}', city_cluster_name, city_for_training, city_for_predicting, technique)
            plt.close()
    
    def showR2ScatterPlots(self, is_model, true_values_dict, predicted_values_dict, city_cluster_name, city_for_training, city_for_predicting, technique):
        for data_portion_type in Plotter.METRICS_PORTIONS_CENTRAL if is_model else Plotter.METRICS_PORTIONS_BORDERING:
            plt.scatter(true_values_dict[data_portion_type], predicted_values_dict[data_portion_type], label = 'R²')
            
            # Generates a single line by creating `x_vals`, a sequence of 100 evenly spaced values between the min and max values in true_values
            flattened_values = np.ravel(true_values_dict[data_portion_type])
            x_vals = np.linspace(min(flattened_values), max(flattened_values), 100)
            plt.plot(x_vals, x_vals, color='red', label='x=y')  # Line will only appear once
            
            plt.title (f'Model {city_for_training} applied to {city_for_predicting}\nR² {data_portion_type} data ({technique})')
            plt.xlabel('True values')
            plt.ylabel('Predicted values')
            plt.legend()
                
            self._saveFig(plt, f'R² Scatter Plot {data_portion_type}', city_cluster_name, city_for_training, city_for_predicting, technique)
            plt.close()