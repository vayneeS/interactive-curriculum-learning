#import scipy.interpolate
import os
import pickle
import seaborn as sns
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pylab as plt

import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd


sel_conditions = ['mabuni', 'scheduled', 'choilike']

def icf(data, diameter=800):
    b1 = diameter / 2 + 9
    b2 = diameter / 2 - 9
    data_ = np.sqrt(np.sum(np.power(data - [500, 500],2), 1))
    outs1 = np.where(data_ > b1)[0]
    outs2 = np.where(data_ < b2)[0]
    outs = list(outs1) + list(outs2)
    return (len(data_) - len(outs))/len(data_)

def compute_metric_pre_post(metric='jerk'):
    results = {'phase': [], 'schedule': [], metric: []}
    bound_min = 5
    bound_max = 95
    for k in ['pretest','posttest']:
        data_ = pickle.load(open('{}.pkl'.format(k), 'rb'))
        for p_id in data_.keys():
            if data_[p_id]['algo'] in sel_conditions:
                for b in data_[p_id]['data'].keys():
                    metric_vals = []
                    for t in data_[p_id]['data'][b].keys():
                        xy = data_[p_id]['data'][b][t]
                        b1 = int(bound_min/100.0 * len(xy))
                        b2 = int(bound_max/100.0 * len(xy))
                        xy = xy[b1:b2,:]
                        if metric == 'jerk':
                            sgxy = [savgol_filter(xy[:,0], 7, 3, deriv=3), savgol_filter(xy[:,1], 7, 3, deriv=3)]
                            jerk = np.sum(np.sum(np.square(sgxy), axis=1), axis=0)
                            if jerk < 2000:
                                metric_vals.append(jerk)
                        elif metric == 'icf':
                            metric_vals.append(icf(xy))
                        elif metric == 'time':
                            metric_vals.append(len(xy))
                    results['phase'].append(k)
                    results['schedule'].append(data_[p_id]['algo'])
                    results[metric].append(np.mean(metric_vals))
    return results

def compute_icf():
    data_ = pickle.load(open('training_xy.pkl', 'rb'))
    data_width = pickle.load(open('training_width.pkl', 'rb'))
    results = {'icf': [], 'schedule': [], 'width' : [], 'participant' : [], 'block_id' : []}
    bound_min = 5
    bound_max = 95
    for p_id in data_.keys():
        if data_[p_id]['algo'] in sel_conditions:
            # metric_vals = []
            for b in data_[p_id]['data'].keys():
                metric_vals = []
                for t in data_[p_id]['data'][b].keys():
                    xy = data_[p_id]['data'][b][t]
                    # xy = xy[int(0.1 * len(xy)):int(0.9 * len(xy)),:]
                    b1 = int(bound_min/100.0 * len(xy))
                    b2 = int(bound_max/100.0 * len(xy))
                    xy = xy[b1:b2,:]
                    metric_vals.append(icf(xy,data_width[p_id]['width'][b][0]))  
                results['participant'].append(p_id)
                results['schedule'].append(data_[p_id]['algo'])
                results['icf'].append(np.mean(metric_vals))
                results['width'].append(data_width[p_id]['width'][b][0])
                results['block_id'].append(b)
    return results

def compute_lp_per_block(df,metric):#
    diff = {}
    blocks = {}
    df_group = df.groupby(['participant','width']) 
    for name, group in df_group:
        if (not name[0] in diff.keys()):
            diff[name[0]] = {}
            blocks[name[0]] = {}
        if(metric == 'icf'):
            diff[name[0]][name[1]] = np.abs(np.diff(group[metric]))
        else:
            diff[name[0]][name[1]] = np.diff(group[metric])
        blks = group['block_id'].values
        blocks[name[0]][name[1]] = blks        
    lp = {}
    lp_lst = []
    widths = []
    blks = []
    schedule = []
    for k in diff.keys():
        lp[k] = {}
        for w in diff[k].keys():
            lp_lst.append(0)
            schedule.append(k.split("-",1)[1])
            for i in range(len(diff[k][w])):
                lp[k][blocks[k][w][i+1]] = diff[k][w][i]
                lp_lst.append(diff[k][w][i])
                schedule.append(k.split("-",1)[1])
            blks.extend(blocks[k][w])
            widths.extend(len(blocks[k][w])*[w])

    df_lp = pd.DataFrame(list(zip(widths, blks, lp_lst, schedule)),
               columns =['Width', 'Block', 'Learning_Progress','Schedule'])
    return df_lp

def compute_metric_retention_transfer(metric='jerk'):
    results = {'phase': [], 'schedule': [], metric: []}
    bound_min = 5
    bound_max = 95
    data_ = pickle.load(open('retention_transfer_xy.pkl', 'rb'))
    for p_id in data_.keys():
        if data_[p_id]['algo'] in sel_conditions:
            for du in data_[p_id]['data'].keys():
                for di in data_[p_id]['data'][du].keys():
                    for b in data_[p_id]['data'][du][di].keys():
                        metric_vals = []
                        for t in data_[p_id]['data'][du][di][b].keys():
                            xy = data_[p_id]['data'][du][di][b][t]
                            # xy = xy[int(0.1 * len(xy)):int(0.9 * len(xy)),:]
                            b1 = int(bound_min/100.0 * len(xy))
                            b2 = int(bound_max/100.0 * len(xy))
                            xy = xy[b1:b2,:]
                            if metric == 'jerk':
                                sgxy = [savgol_filter(xy[:,0], 7, 3, deriv=3), savgol_filter(xy[:,1], 7, 3, deriv=3)]
                                jerk = np.sum(np.sum(np.square(sgxy), axis=1), axis=0) 
                                metric_vals.append(jerk )
                            elif metric == 'icf':
                                metric_vals.append(icf(xy, diameter=di))
                            elif metric == 'time':
                                metric_vals.append(len(xy))
                        results['phase'].append('{}-{}'.format(du,di))
                        results['schedule'].append(data_[p_id]['algo'])
                        results[metric].append(np.mean(metric_vals))
    return results

def hist_participant(data):
    fig, ax = plt.subplots(6,int(len(data.keys())/6),figsize=(15,15))

    c=0
    r=0
    for k in data.keys():
        res = {'algo': [], 'width': []}
        for b in data[k]['width'].keys():
            res['algo'].append(data[k]['algo'])
            res['width'].append(data[k]['width'][b][0])
        df = pd.DataFrame(res)

        sns.histplot(data=df, x="width", bins=7, ax = ax[r,c])
        ax[r,c].title.set_text(k)

        c+=1
        if(c>int(len(data.keys())/6)-1):
            c=0
            r+=1 
        fig.tight_layout()
    plt.show()

def plot_mean_hist(data):
    #res = {'algo': [], 'width': [],'participant':[]}
    res = {'width': [],'participant':[]}
    for k in data.keys():
        if("MAB" in k):
            for b in data[k]['width'].keys():
                res['participant'].append(k)
                #res['algo'].append(data[k]['algo'])
                width = data[k]['width'][b][0]
                res['width'].append(width)

    df_width = pd.DataFrame(res)
    #df2 = df_width.groupby(['algo','width','participant']).agg({'width':'size'}).rename(columns={'width':'count'})
    df2 = df_width.groupby(['width','participant']).agg({'width':'size'}).rename(columns={'width':'count'})
    df2=df2.reset_index()
    
    model = ols('count ~ C(width)', data=df2).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    #print(df2)
    sns.boxplot(data=df2,x=df2['width'],y=df2['count']).set_title("mean task distribution")
    sns.stripplot(data=df2,x=df2['width'],y=df2['count'],dodge=True)
    plt.show()

def plot_mt(data,phase):

    dic_len_xy = {'algo' : [],'len_xy' : [],'block' : []}

    for k in data.keys():

        xy_block = []
        for i in data[k]['data'].keys():
            xy_trial = 0
            for t in data[k]['data'][i].keys():
                xy_trial+=len(data[k]['data'][i][t])
            xy_block.append(xy_trial/len(data[k]['data'][i].keys()))
            dic_len_xy['block'].append(i)
        dic_len_xy['algo'].extend(len(xy_block)*[data[k]['algo']])
        dic_len_xy['len_xy'].extend(xy_block)

    df_len_xy = pd.DataFrame(dic_len_xy)
    
    sns.boxplot(data=df_len_xy,x='block',y='len_xy',hue='algo').set_title(phase)
    sns.stripplot(data=df_len_xy,x='block',y='len_xy',hue='algo',dodge=True)
    plt.show()

def plot_mt_ret_trans(data_ret_trans,phase):

    dic_len_xy = {'algo' : [],'len_xy' : [],'block' : [], 'phase' : []}
       
    for p_id in data_ret_trans.keys():
        xy_block = []
        if data_ret_trans[p_id]['algo'] in sel_conditions:
            for du in data_ret_trans[p_id]['data'].keys():
                for di in data_ret_trans[p_id]['data'][du].keys():
                    for b in data_ret_trans[p_id]['data'][du][di].keys():
                        
                        xy_trial = 0
                        for t in data_ret_trans[p_id]['data'][du][di][b].keys():
                            xy_trial+=len(data_ret_trans[p_id]['data'][du][di][b][t])
                        xy_block.append(xy_trial/len(data_ret_trans[p_id]['data'][du][di].keys()))
                        dic_len_xy['block'].append(b)
                        dic_len_xy['phase'].append('{}-{}'.format(du,di))
            dic_len_xy['algo'].extend(len(xy_block)*[data_ret_trans[p_id]['algo']])
            dic_len_xy['len_xy'].extend(xy_block)

    df_len_xy = pd.DataFrame(dic_len_xy)
    if phase == "retention":
        df_new = df_len_xy[df_len_xy['phase'] == '1.0-800']
        sns.boxplot(data=df_new,x='block',y='len_xy',hue='algo').set_title(phase)
        sns.stripplot(data=df_new,x='block',y='len_xy',hue='algo',dodge=True)
        
    else:
        df_new = df_len_xy[df_len_xy['phase'] != '1.0-800']
        sns.boxplot(data=df_new,x='phase',y='len_xy',hue='algo').set_title(phase)
        sns.stripplot(data=df_new,x='phase',y='len_xy',hue='algo',dodge=True)
    plt.show()
    
def plot_icf_jerk(data1,data2):
    sns.scatterplot(x=data1,y=data2)
    plt.show()

#------------------#############################################-------------------------#
# results_icf = compute_icf()
# df_icf=pd.DataFrame(results_icf)
# df_lp_icf = compute_lp_per_block(df_icf,'icf')
# fig, ax = plt.subplots(figsize=(13,7))
# ax = sns.scatterplot(data=df_lp_icf,x="Block",y="Learning_Progress",hue="Width",style="Schedule")

# metric = 'jerk'
metric = 'icf'

data_training = pickle.load(open('training_width.pkl', 'rb'))
#hist_participant(data_training)

results_pre_post = compute_metric_pre_post(metric)
df_pre_post = pd.DataFrame(results_pre_post)
# df_pre = df_pre_post.loc[df_pre_post['phase'] == "pretest"]
# df_post = df_pre_post.loc[df_pre_post['phase'] == "posttest"]

results_ret_transf = compute_metric_retention_transfer(metric=metric)
df_ret_transf = pd.DataFrame(results_ret_transf)

df_ret = df_ret_transf.loc[df_ret_transf['phase'] == '1.0-800']
df_ret['phase'] = 'retention'

df_all = pd.concat([df_pre_post,df_ret], axis=0)
for phase in (df_ret_transf['phase'].unique()):
    if(phase != '1.0-800'):
        print(phase)
        df_trans = df_ret_transf.loc[df_ret_transf['phase'] == phase]
        df_all = pd.concat([df_all,df_trans], axis=0)

# print(df_all)    
# plt.figure(figsize=(8,5))
# sns.boxplot(x="phase", y=metric, hue="schedule", data=df_all, palette="Blues")
#plt.show()

model = ols('{} ~ C(phase) + C(schedule) + C(phase):C(schedule)'.format(metric), data=df_all).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
# print(anova_table)

#print("test pre retention: ",ttest_ind(df_pre[metric], df_ret[metric]))

#phase and schedule

###
#t-test 
# data_ret_trans = pickle.load(open('retention_transfer_xy.pkl', 'rb'))

# data_posttest = pickle.load(open('posttest.pkl', 'rb'))
# data_pretest = pickle.load(open('pretest.pkl', 'rb'))

# plot_mt_ret_trans(data_ret_trans,"transfer")
# plot_mt_ret_trans(data_ret_trans,"retention")
# plot_mt(data_posttest,"posttest")
# plot_mt(data_pretest,"pretest")
plot_mean_hist(data_training)


#plot_icf_jerk(results_pre_jerk['jerk'],results_pre_icf['icf'])
# covariance = np.cov(results_pre_jerk['jerk'], results_pre_icf['icf'])
# print(covariance)

# cor_pearson,_ = pearsonr(results_pre_jerk['jerk'],results_pre_icf['icf'])
# print(cor_pearson)