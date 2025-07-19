
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns


class DataLoader:

    def load_data(self, folder_name='../experiments/Training'):
        data = {}
        for root, _, files in os.walk(folder_name):
            for f in files:
                if f.endswith('.db'):
                    p_id = os.path.basename(root)
                    data[p_id] = pd.read_json(os.path.join(root, f), lines=True)
        return data

    def filter_blocks(self, data, lower, upper):
        filtered = {}
        for k, df in data.items():
            filtered[k] = df[(df['icf'] != -1) & (df['block_id'] > lower) & (df['block_id'] <= upper)]
        return filtered

    def training(self, data):
        return self.filter_blocks(data, lower=3, upper=94)

    def pre_test(self, data):
        return self.filter_blocks(data, lower=0, upper=1)

    def post_test(self, data):
        return self.filter_blocks(data, lower=95, upper=96)

    def icf_per_block(self,data,p_id):
        return data[p_id].groupby(['block_id']).mean()['icf']

    def get_data(self,condition,data):
        d = {}       
        for k in data.keys():
            if(condition in k):
                d[k] = data[k]
        return d

    def prepare_df(self, data, jerk_data, condition):
    """
    Prepare a DataFrame of jerk values with condition labels for analysis or plotting.
    
    Parameters:
        data (dict): participant-wise metadata
        jerk_data (dict): raw trajectory data
        condition (str or int): label for experimental condition
    
    Returns:
        DataFrame: DataFrame with 'Jerk' and 'Condition' columns
    """
    pretest_vals = self.jerk_per_block(data, jerk_data, block=4)

    all_jerk_vals = [val for lst in pretest_vals.values() for val in lst]

    condition_labels = np.full(len(all_jerk_vals), condition)

    df = pd.DataFrame({
        "Jerk": all_jerk_vals,
        "Condition": condition_labels
    })
    return df

    def prepare_icf_df(self, test_val, condition):
        """
        Prepare ICF data and associate each value with an experimental condition.
        
        Parameters:
            test_val (dict): {participant_id: list of ICF values}
            condition (str or int): condition label (e.g., 'MAB', 'Random', 'Choi')
        
        Returns:
            pd.DataFrame: DataFrame with columns 'Icf' and 'Condition'
        """
        all_icf_vals = [val for lst in test_val.values() for val in lst]

        condition_labels = np.full(len(all_icf_vals), condition)

        df = pd.DataFrame({
            "Icf": all_icf_vals,
            "Condition": condition_labels
        })
        return df

    def icf_slope(self, data, ax):
        condition_map = {'MAB': 0, 'Random': 1, 'Choi': 2}
        participant_counts = {0: 0, 1: 0, 2: 0}

        for k, df in data.items():
            grouped = df.groupby('block_id')['icf'].mean().reset_index()
            x = grouped['block_id'].values
            y = np.log(grouped['icf'].values)

            slope, intercept, _, _, pvalue = stats.linregress(x, y)
            y_fit = slope * x + intercept

            # Determine condition index
            condition_idx = 2  # default to Choi
            for cond, idx in condition_map.items():
                if cond in k:
                    condition_idx = idx
                    break

            col = participant_counts[condition_idx]

            ax[condition_idx, col].plot(x, y_fit, 'r', label=f'p={pvalue:.3f}')
            ax[condition_idx, col].plot(x, y, 'o')
            ax[condition_idx, col].set_title(k)
            ax[condition_idx, col].set_xlabel("block")
            ax[condition_idx, col].legend()

            participant_counts[condition_idx] += 1

        # Set y-axis label
        for row in range(3):
            ax[row, 0].set_ylabel("log icf")

    def plot_task_distribution(self, data_mab, data_random, data_choi):
        condition_data = {'MAB': data_mab, 'Random': data_random, 'Choi': data_choi}
        max_participants = max(len(d) for d in condition_data.values())

        fig, axes = plt.subplots(nrows=3, ncols=max_participants, figsize=(20, 20), squeeze=False)

        for i, (label, data) in enumerate(condition_data.items()):
            for j, (participant, df) in enumerate(data.items()):
                axes[i, j].plot(df['block_id'], df['width'], marker='o', linestyle='dashed')
                axes[i, j].set_title(f'{label} - P{participant}')
            # Hide empty plots
            for j in range(len(data), max_participants):
                axes[i, j].axis('off')

    fig.tight_layout()

    def plot_jerk_slope(self, data, jerk_norm, ax):
        jerk_blocks = self.jerk_per_block(data, jerk_norm, 4)
        condition_map = {'MAB': 0, 'Random': 1, 'Choi': 2}
        participant_counts = {0: 0, 1: 0, 2: 0}

        for k, jerk_values in jerk_blocks.items():
            y = np.log(jerk_values)
            x = np.arange(len(y))
            slope, intercept, _, _, pvalue = stats.linregress(x, y)
            y_fit = slope * x + intercept

            # Determine condition index
            condition_idx = 2  # default
            for cond, idx in condition_map.items():
                if cond in k:
                    condition_idx = idx
                    break
            col = participant_counts[condition_idx]

            ax[condition_idx, col].plot(x, y_fit, 'r', label=f'p={pvalue:.3f}')
            ax[condition_idx, col].plot(x, y, 'o')
            ax[condition_idx, col].set_title(k)
            ax[condition_idx, col].set_xlabel("block")
            ax[condition_idx, col].legend()

            participant_counts[condition_idx] += 1
        # Set y-labels once per condition row
        for row in range(3):
            ax[row, 0].set_ylabel("log jerk")
    
    def jerk_norm(self,data,start_idx,stop_idx,num_trials):    
        ''' Computes a normalized jerk metric for each participant over a range of trials and blocks
            Uses Savitzky–Golay filter to estimate jerk per trial
            start_idx: starting block
            stop_idx: last block
            num_trials: number of trials to group jerk
        '''    
        px = {}
        py = {}
        for k in data.keys():
            df = data[k]
            df_=df[(df['block_id'] > start_idx) & (df['block_id'] <= stop_idx) & (~df['y'].isnull())]
            temp = df_.loc[:,['x','y','block_id','trial_id']].copy()
            temp['y_'] = 1000 - df_['y']
            px[k] = []
            py[k] = []
            for b in range(start_idx + 1,stop_idx + 1):
                for i in range(num_trials):
                    x = df_[(df_['trial_id'] == i) & (df_['block_id'] == b)]['x']
                    y = temp[(temp['trial_id'] == i) & (temp['block_id'] == b)]['y_']
                    sgx = savgol_filter(x,7,3,deriv=3)
                    sgy = savgol_filter(y,7,3,deriv=3)
                    px[k].append(sgx)
                    py[k].append(sgy)
        jksq = {}
        for k in px.keys():
            jksq[k] = []
            for it in range(len(px[k])):
                jk = 0
                zipped = list(zip(px[k][it],py[k][it]))
                jk += np.linalg.norm(zipped,2)
                n = len(px[k][it])
                jksq[k].append(jk/n)
        return jksq
    
    def jerk_stats_transfer(self, data):
    """
    Compute mean and standard deviation of jerk for each participant.
    
    Parameters:
        data (dict): {participant_id: list of jerk values}
    
    Returns:
        tuple: (dict of means, dict of std deviations)
    """
    grp_avg = {k: np.mean(v) for k, v in data.items()}
    grp_std = {k: np.std(v) for k, v in data.items()}
    return grp_avg, grp_std

