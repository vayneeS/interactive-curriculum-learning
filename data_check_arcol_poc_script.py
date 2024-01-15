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
import json

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

def get_data_phases(folder_name = 'Training',path = './experiments/'):
        path = path + folder_name
        data = {}
        data_trials =  {}
        for root,dir, files in os.walk(path):
            for file in files:
                if file.endswith(".db"):
                    p_id = root.split('/')[-1] 
                    data[p_id] = {}
                    data_trials[p_id] = {}
                    with open(os.path.join(root, file)) as f:
                        db = f.read().splitlines()
                        data[p_id]['data'] = {}
                        for j in range(len(db)):
                            line_db = json.loads(db[j])
                            if j == 0 : data[p_id]['algo'] = line_db['algo']

                            if ('x' in line_db.keys()):
                                if(line_db['block_id'] not in data_trials[p_id].keys()):
                                    data_trials[p_id][line_db['block_id']] = {}
                                if(line_db['trial_id'] not in data_trials[p_id][line_db['block_id']].keys()):
                                    data_trials[p_id][line_db['block_id']][line_db['trial_id']] = []
                                data_trials[p_id][line_db['block_id']][line_db['trial_id']].append([line_db['x'],line_db['y']])
                                
                                for blk_id in range(97):
                                    if blk_id not in data[p_id]['data'].keys():
                                        data[p_id]['data'][blk_id] = {}
                                    for t_id in range(4):
                                        if t_id not in data[p_id]['data'][blk_id].keys():
                                            data[p_id]['data'][blk_id][t_id] = []
        for pid in data.keys():
            for b in data[pid]['data'].keys():
                for t in data[pid]['data'][b].keys():
                    data[pid]['data'][b][t] = np.array(data_trials[pid][b][t])
        # print(data['P012-MAB']['data'])

        return data

def pickle_ret_transfer(folder_name = 'Test',path = './experiments/',filename = 'retention_transfer_xy.pkl'):
    path = path + folder_name
    data = {}
    data_trials = {}

    for root,dir, files in os.walk(path):
        for file in files:
            if file.endswith(".db"):
                p_id = root.split('/')[-1] 
                data[p_id] = {}
                data_trials[p_id] = {}
                with open(os.path.join(root, file)) as f:
                    db = f.read().splitlines()
                    data[p_id]['data'] = {}
                    for j in range(len(db)):
                        line_db = json.loads(db[j])

                        if j == 0 : 
                            if 'MAB' in p_id:
                                algo = 'mabuni'
                            elif 'Choi' in p_id:
                                algo = 'choilike'
                            elif 'Random' in p_id:
                                algo = 'scheduled'

                            data[p_id]['algo'] = algo
                        
                        if ('x' in line_db.keys()):
                            if line_db['block_id'] not in data_trials[p_id].keys():
                                data_trials[p_id][line_db['block_id']] = {}
                            if line_db['trial_id'] not in data_trials[p_id][line_db['block_id']].keys():
                                data_trials[p_id][line_db['block_id']][line_db['trial_id']] = []
                            data_trials[p_id][line_db['block_id']][line_db['trial_id']].append([line_db['x'],line_db['y']])
                            
                            for i in [1.0,0.7]:
                                if(line_db['mouvement_time']/1000 == i):
                                    if i not in data[p_id]['data'].keys():
                                        data[p_id]['data'][i] = {}
                                    for l in [400,600,800]:
                                        if(l not in data[p_id]['data'][i].keys()):
                                            data[p_id]['data'][i][l] = {}
                                        if(l == 800):
                                            blk_ids = [0,1,4]
                                        elif(l == 600):
                                            blk_ids = [2,5]   
                                        elif(l == 400):
                                            blk_ids = [3,6]
                                        for blk_id in blk_ids:
                                            if blk_id not in data[p_id]['data'][i][l].keys():
                                                data[p_id]['data'][i][l][blk_id] = {}
                                                for t_id in range(12):
                                                    if t_id not in data[p_id]['data'][i][l][blk_id].keys():
                                                        data[p_id]['data'][i][l][blk_id][t_id] = []
                                       
    for pid in data.keys():
        for mt in data[pid]['data'].keys():
            for l in data[pid]['data'][mt].keys():
                for b in data[pid]['data'][mt][l].keys():
                    for t in data[pid]['data'][mt][l][b].keys():
                        data[pid]['data'][mt][l][b][t] = np.array(data_trials[pid][b][t])
                                        
    # print(data['P012-MAB'])
    # print(data)
    file = open('./pickles/'+filename,'wb')
    pickle.dump(data,file)
            
def pickle_phases(data,filename):
    new_data = {}

    for pid in data.keys():
        new_data[pid] = {}
        new_data[pid]['algo'] = data[pid]['algo']
        new_data[pid]['data'] = {}
        if(filename == 'pretest_xy.pkl'):
            for blk_id in data[pid]['data'].keys():
                if(blk_id > 0 and blk_id <=3):
              
                    new_data[pid]['data'][blk_id] = data[pid]['data'][blk_id]
        elif(filename == 'posttest_xy.pkl'):
            for blk_id in data[pid]['data'].keys():
                if(blk_id > 94 and blk_id <=97):
      
                    new_data[pid]['data'][blk_id] = data[pid]['data'][blk_id]
        elif(filename == 'training_xy.pkl'):
            for blk_id in data[pid]['data'].keys():
                if(blk_id >3 and blk_id <=94):
           
                    new_data[pid]['data'][blk_id] = data[pid]['data'][blk_id]
    print(new_data)
    file = open('./pickles/'+filename,'wb')
    pickle.dump(new_data,file)

def compute_std_jerk(map_algo,map_phases,data_post,data_rettrasnf,bounds=[0., 1.]):
    
    jerk_trials = {'condition':[],'jerk':[],'phase':[]}
    data_post_ret_trans = {}
    data_post_ret_trans['D1-1.0-800-Post'] = data_post

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
                data_post_ret_trans['D2-{}-{}-Ret'.format(mt, dm)] = dat
            else:
                data_post_ret_trans['D2-{}-{}-Tsf'.format(mt, dm)] = dat
    jerk_trials = {'phase':[],'condition':[],'jerk':[]}
    for k in data_post_ret_trans.keys(): 
        # print(k)
        # data_ = pickle.load(open(os.path.join(pickle_folder, '{}_xy.pkl'.format(k)), 'rb'))
        data_ = data_post_ret_trans[k]
        # loop on participants
        for p_id in data_.keys():
            # loop on blocks
            for b in data_[p_id]['data'].keys():
               
                for t in data_[p_id]['data'][b].keys():
                    if '{}_{}_{}_{}'.format(k, p_id, b, t) != 'D1-1.0-800-Pre_P1-MAB_1_1':
                        xy = data_[p_id]['data'][b][t]
                        b1 = int(bounds[0] * len(xy))

                        b2 = int(bounds[1] * len(xy))
                        xy = xy[b1:b2, :]
                        
                        if jerk(xy) < 1200:
                            jerk_trial = jerk(xy)

                        jerk_trials['jerk'].append(np.mean(jerk_trial))
                        jerk_trials['condition'].append(map_algo[data_[p_id]['algo']])
                        jerk_trials['phase'].append(map_phases[k])
   
    #removing outliers of jerk data for each phase and condition
    filtered_jerk = {'phase':[],'jerk':[],'condition':[]}
    for phase in np.unique(jerk_trials['phase']):
        for cond in np.unique(jerk_trials['condition']):
            idx1 = np.where(np.array(jerk_trials['condition']) == cond)[0]
            idx2 = np.where(np.array(jerk_trials['phase']) == phase)[0]
            idx = list(set(idx1) & set(idx2))
            
            sel_data = np.array(jerk_trials['jerk'])[idx]
            mean_distrib = np.mean(np.array(jerk_trials['jerk'])[idx])
            outlier_thresh = mean_distrib + 3 * np.std(np.array(jerk_trials['jerk'])[idx])
            idx_no_outliers = np.where(sel_data < outlier_thresh)[0]
            for i in idx_no_outliers:
  
                filtered_jerk['phase'].append(phase)
                filtered_jerk['condition'].append(cond)
                filtered_jerk['jerk'].append(sel_data[i])

    std_per_block = {'condition':[],'phase':[],'std':[]}
    for phase in np.unique(filtered_jerk['phase']):
        for cond in np.unique(filtered_jerk['condition']):
            idx0 = np.where(np.array(filtered_jerk['condition']) == cond)[0]
            idx1 = np.where(np.array(filtered_jerk['phase']) == phase)[0]
            idx = list(set(idx0) & set(idx1))
            
            std_per_block['std'].append(np.std(np.array(filtered_jerk['jerk'])[idx]))
            std_per_block['condition'].append(cond)
            std_per_block['phase'].append(phase)

    df = pd.DataFrame(std_per_block)
    return df     

def plot_std_jerk(df):
    order = ['Error Adaptation','Curriculum Learning','Random']
    ax = sns.barplot(
    x='condition',
    y='std',
    hue_order = ['Error Adaptation','Curriculum Learning','Random'],
    hue='condition',
    data=df,
    palette="Set2",
    ci="sd", 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.1,
    order=order)
    ax.legend_.remove()
    ax.tick_params(axis='x', rotation=25)
    ax.set(xlabel=None)
    plt.ylabel('JERK standard deviation', fontsize=17)
    # plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
    # plt.setp(ax.get_legend().get_title(), fontsize='17')
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tight_layout()
    plt.savefig('jerk_std_plot.pdf')
    plt.show()  

if __name__ == "__main__":
    pickle_folder = './pickles'
    map_algo = {'choilike':'Error Adaptation','mabuni':'Curriculum Learning','scheduled':'Random'}
    map_phases = {'D1-1.0-800-Pre':'Pre-test', 
            'D1-1.0-800-Post':'Post-test', 
            'D2-1.0-800-Ret':'Retention', 
            'D2-1.0-600-Tsf':'Transfer-1', 
            'D2-1.0-400-Tsf':'Transfer-2', 
            'D2-0.7-800-Tsf':'Transfer-3',
            'D2-0.7-600-Tsf':'Transfer-4',
            'D2-0.7-400-Tsf':'Transfer-5'}
    
    data_pre = pickle.load(open(os.path.join(pickle_folder, 'pretest_xy.pkl'), 'rb'))
    data_post = pickle.load(open(os.path.join(pickle_folder, 'posttest_xy.pkl'), 'rb'))
    data_rettrasnf = pickle.load(open(os.path.join(pickle_folder, 'retention_transfer_xy.pkl'), 'rb'))
    df_jerk_var = compute_std_jerk(map_algo,map_phases,data_post,data_rettrasnf)
    plot_std_jerk(df_jerk_var)