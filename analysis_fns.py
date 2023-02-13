import os
import pickle
import numpy as np
from scipy.signal import savgol_filter

import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns

from scipy import stats

import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
from scipy.stats import ttest_ind
from manuals import get_manual_cuts

manual_cuts = get_manual_cuts()

def icf(data, width=18, diameter=400):
    data_ = np.sqrt(np.sum(np.power(data - [500, 500], 2), 1))
    outs1 = np.where(data_ > diameter + width/2)[0]
    outs2 = np.where(data_ < diameter - width/2)[0]
    outs = list(outs1) + list(outs2)
    return (len(data_) - len(outs))/len(data_)


def jerk(data):
    sgxy = [
        savgol_filter(data[:, 0], 7, 3, deriv=3), 
        savgol_filter(data[:, 1], 7, 3, deriv=3)]
    return np.sum(
        np.sum(np.square(sgxy), axis=1), axis=0)
    

def performance(pickle_folder='./',
                phases=['D1 1.0s 800px (Pre)', 'D1 1.0s 800px (Post)'],
                metrics=['icf', 'jerk', 'mvt_time'],
                bounds=[0., 1.],
                save=False):
    
    results = {'phase': [], 'schedule': []}
    for m in metrics:
        results[m] = []

    data_pre = pickle.load(open(os.path.join(pickle_folder, 'pretest_xy.pkl'), 'rb'))
    data_post = pickle.load(open(os.path.join(pickle_folder, 'posttest_xy.pkl'), 'rb'))
    data_rettrasnf = pickle.load(open(os.path.join(pickle_folder, 'retention_transfer_xy.pkl'), 'rb'))

    # phases = ['D1 1.0s 800px (Pre)', 
    #             'D1 1.0s 800px (Post)', 
    #             'D2 1.0s 800px (Ret)', 
    #             'D2 1.0s 600px (Tsf)', 
    #             'D2 1.0s 400px (Tsf)', 
    #             'D2 0.7s 800px (Tsf)',
    #             'D2 0.7s 600px (Tsf)',
    #             'D2 0.7s 400px (Tsf)']
    
    all_data = {}
    all_data['D1-1.0-800-Pre'] = data_pre
    all_data['D1-1.0-800-Post'] = data_post
    for mt in [0.7, 1.0]:
        for dm in [800, 600, 400]:
            dat = {}
            for p_id in data_rettrasnf.keys():
                dat[p_id] = {'algo': data_rettrasnf[p_id]['algo'], 'data': {}}
                bid = 0
                for b in data_rettrasnf[p_id]['data'][mt][dm].keys():
                    dat[p_id]['data'][bid] = {}
                    for t in data_rettrasnf[p_id]['data'][mt][dm][b].keys():
                        dat[p_id]['data'][bid][t] = data_rettrasnf[p_id]['data'][mt][dm][b][t]
                    bid += 1
            if mt == 1.0 and dm == 800:
                all_data['D2-{}-{}-Ret'.format(mt, dm)] = dat
            else:
                all_data['D2-{}-{}-Tsf'.format(mt, dm)] = dat
                # print('D2 {}s {}px (Tsf)'.format(mt, dm))
                # print(mt, dm, all_data['D2 {}s {}px (Tsf)'.format(mt, dm)])

    print(all_data.keys())

    # get data and compute metrics
    for k in phases: 
        # data_ = pickle.load(open(os.path.join(pickle_folder, '{}_xy.pkl'.format(k)), 'rb'))
        data_ = all_data[k]
        # loop on participants
        for p_id in data_.keys():
            # loop on blocks
            for b in data_[p_id]['data'].keys():
                metric_vals = {}
                for m in metrics:
                    metric_vals[m] = []
                # loop on trials
                for t in data_[p_id]['data'][b].keys():
                    if '{}_{}_{}_{}'.format(k, p_id, b, t) != 'D1-1.0-800-Pre_P1-MAB_1_1':
                        xy = data_[p_id]['data'][b][t]
                        b1 = int(bounds[0] * len(xy))
                        # if '{}_{}_{}_{}'.format(k, p_id, b, t) in manual_cuts.keys():
                        #     b2 = manual_cuts['{}_{}_{}_{}'.format(k, p_id, b, t)]
                        # else:
                        b2 = int(bounds[1] * len(xy))
                        xy = xy[b1:b2, :]
                        if 'jerk' in metrics:
                            if jerk(xy) < 1200:
                                metric_vals['jerk'].append(jerk(xy))
                        if 'icf' in metrics:
                            diam = int(k.split('-')[-2]) / 2
                            metric_vals['icf'].append(icf(xy, width=18, diameter=diam))
                        if 'mvt_time' in metrics:
                            metric_vals['mvt_time'].append(len(xy) / 200.0)
                results['phase'].append(k)
                results['schedule'].append(data_[p_id]['algo'])  
                for met in metric_vals.keys():
                    # results[met].append(metric_vals[met][-1])
                    results[met].append(np.mean(metric_vals[met]))

    # # remove outliers (above 3 * std)
    # levels = [[n for n in np.unique(results[f])] for f in ['phase', 'schedule']]
    # results_per_metrics = {}
    # for met in metric_vals.keys():
    #     results_per_metrics[met] = {'phase': [], 'schedule': [], '{}'.format(met): []}
    #     for l1 in levels[0]:
    #         for l2 in levels[1]:
    #             idx1 = np.where(np.array(results['phase']) == l1)[0]
    #             idx2 = np.where(np.array(results['schedule']) == l2)[0]
    #             idx = list(set(idx1) & set(idx2))
    #             sel_data = np.array(results[met])[idx] 
    #             mean_distrib = np.mean(np.array(results[met])[idx])
    #             outlier_thresh = mean_distrib + 3 * np.std(np.array(results[met])[idx])
    #             idx_noo = np.where(sel_data < outlier_thresh)[0]
    #             for i in idx_noo:
    #                 results_per_metrics[met]['phase'].append(l1)
    #                 results_per_metrics[met]['schedule'].append(l2)
    #                 results_per_metrics[met][met].append(sel_data[i])
    
    results_per_metrics = {}
    for met in metric_vals.keys():
        results_per_metrics[met] = {'phase': [], 'schedule': [], '{}'.format(met): []}
        idx = np.arange(len(results['phase']))
        for i in idx:
            results_per_metrics[met]['phase'].append(results['phase'][i])
            results_per_metrics[met]['schedule'].append(results['schedule'][i])
            results_per_metrics[met][met].append(results[met][i])

    if save:
        pickle.dump(results, open('results_per_metrics.pkl', 'wb'))
    return results_per_metrics


def learning_rates_per_width(pickle_folder='./',
                   width=25,
                   metrics=['icf', 'jerk'],
                   bounds=[0., 1.],
                   save=False):
    
    results = {'schedule': [], 'width': []}
    for m in metrics:
        results[m] = []
    
    data_ = pickle.load(open(os.path.join(pickle_folder, 'training_xy.pkl'), 'rb'))
    width_ = pickle.load(open(os.path.join(pickle_folder, 'training_width.pkl'), 'rb'))

    widths = [25, 30, 35, 40, 45, 50, 55]
    
    for p_id in data_.keys():
        if data_[p_id]['algo'] in ['mabuni', 'scheduled', 'choilike']:

            metric_vals = {}
            times = {}
            counters = {}
            
            for m in metrics:
                metric_vals[m] = []
                times[m] = []
                counters[m] = 0
            
            metric_per_width = {}
            
            for w in widths:
                metric_per_width[w] = {}
                for m in metrics:
                    metric_per_width[w][m] = []

            for b in data_[p_id]['data'].keys():
                for t in data_[p_id]['data'][b].keys():
                    w = width_[p_id]['width'][b][t]
                    xy = data_[p_id]['data'][b][t]
                    b1 = int(bounds[0] * len(xy))
                    b2 = int(bounds[1] * len(xy))
                    xy = xy[b1:b2, :]
                    if 'jerk' in metrics:
                        metric_per_width[w]['jerk'].append(jerk(xy))
                    if 'icf' in metrics:
                        metric_per_width[w]['icf'].append(icf(xy, width=18, diameter=400))
            
            for w in widths:
                results['schedule'].append(data_[p_id]['algo'])  
                results['width'].append(w)  
                for m in metric_vals.keys():
                    slope, intercept, r, p, se = stats.linregress(np.arange(len(metric_per_width[w][m])), np.log(metric_per_width[w][m]))
                    results[m].append(slope)

    if save:
        pickle.dump(results, open('learning_rates_width.pkl', 'wb'))
    return results



def anova(data, factors=[], measure='', plot=True):
    expr_measure = '{}'.format(measure)
    expr_factors = 'C({})'.format(factors[0]) if len(factors)==1 else 'C({}) + C({}) + C({}):C({})'.format(factors[0], factors[1], factors[0], factors[1])
    model = ols('{} ~ {}'.format(expr_measure, expr_factors), data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # if plot:
    #     ax=sns.boxplot(data=data, x=factors[0], y=measure, hue=factors[1], notch=True, showcaps=False, palette="Set2")
    #     sns.stripplot(data=data, x=factors[0], y=measure, hue=factors[1], palette="Set2", dodge=True, ax=ax, ec='k', linewidth=1, alpha=0.5)
    #     handles, labels = ax.get_legend_handles_labels()
    #     ax.legend(handles[:2], labels[:2], title='Phase', bbox_to_anchor=(1, 1.02), loc='upper left')
    #     print('saving:', 'anova_{}_{}.pdf'.format(expr_measure, expr_factors))
    #     plt.tight_layout()
    #     plt.savefig('anova_{}_{}.pdf'.format(expr_measure, expr_factors), dpi=150)
    #     plt.show()

if __name__ == "__main__":
    pickle_folder = '../pickles'    
    # lw = learning_rates_per_width(pickle_folder=pickle_folder,
    #                          width=25,
    #                          metrics=['icf', 'jerk'],
    #                          bounds=[0., 0.98],
    #                          save=True)
    lw = pickle.load(open('learning_rates_width.pkl', 'rb'))

    df = pd.DataFrame(lw)
    print(len(df))

    anova(data=df, factors=['schedule'], measure='jerk')
    # # print(ttest_ind(df[df['schedule'] == 'mabuni']['jerk'],
    # #         df[df['schedule'] == 'scheduled']['jerk']))
    # # print(ttest_ind(df[df['schedule'] == 'choilike']['jerk'],
    # #         df[df['schedule'] == 'scheduled']['jerk']))
    # # print(ttest_ind(df[df['schedule'] == 'choilike']['jerk'],
    # #         df[df['schedule'] == 'mabuni']['jerk']))

    sns.boxplot(data=df, x='schedule', y='jerk')
    plt.show()
    # results = performance(
    #     pickle_folder=pickle_folder, 
    #     phases=['pretest', 'posttest'],
    #     metrics=['icf', 'jerk', 'mvt_time'], 
    #     bounds=[0.0, 1.], 
    #     save=False)