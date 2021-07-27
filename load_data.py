import os 
import pandas as pd
import numpy as np
from scipy import stats        
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns 

class DataLoader:

    def load_data(self,folder_name = '../experiments/pilote/Training'):
        data = {}
        for root,dir, files in os.walk(folder_name):
            for f in files:
                if f.split('.')[-1] == 'db':
                    p_id = root.split('/')[-1] # P3 or P2
                    data[p_id] = pd.read_json(os.path.join(root, f), lines=True)
        return data

    def training(self,data):
        training = {}
        for k in data.keys():
            df = data[k]
            training[k] = (df[(df['icf'] != -1) & (df['block_id'] > 3) & (df['block_id'] <= 94)])
        return training

    def pre_test(self,data):
        pretest = {}
        for k in data.keys():
            #print(p_id)
            df = data[k]
            pretest[k] = (df[(df['icf'] != -1) & (df['block_id'] > 0) & (df['block_id'] < 4)])
        return pretest

    def post_test(self,data):
        posttest = {}
        for k in data.keys():
            df = data[k]
            posttest[k] = (df[(df['icf'] != -1) & (df['block_id'] >94) & (df['block_id'] < 98)])
        return posttest

    def avg_icf_participant(self,data,p_id):
        return data[p_id].groupby(['block_id']).mean()['icf'].mean()

    def icf_per_block(self,data,p_id):
        return data[p_id].groupby(['block_id']).mean()['icf']

    def mab_data(self,data):
        d = {}       
        for k in data.keys():
            if('MAB' in k):
                d[k] = data[k]
        return d

    def random_data(self,data):
        d = {}        
        for k in data.keys():
            if('Random' in k):
                d[k] = data[k]
        return d

    def choi_data(self,data):
        d = {}
         
        for k in data.keys():
            if('Choi' in k):
                d[k] = data[k]
        return d

    def transfer_data(self,data):
        transfer = {}
        for k in data.keys():
            df = data[k]
            transfer[k] = (df[(df['icf'] != -1) & (df['block_id'] >=2)])
        return transfer

    def retention_data(self,data):
        retention = {}
        for k in data.keys():
            df = data[k]
            retention[k] = (df[(df['icf'] != -1) & (df['block_id'] >=0) & (df['block_id'] <2)])
        return retention
    
    def icf_stats_training_test(self,data):

        lst_avg_participant = []

        for pid in data.keys():
            avg_per_participant = self.avg_icf_participant(data,pid)
            lst_avg_participant.append(avg_per_participant)
        avg = np.mean(lst_avg_participant)
        std = np.std(lst_avg_participant)
        return avg,std

    def icf_stats_transfer(self,data):
        block_avg = {}
        block_std = {}
        dt = data.values()
        iterdt = iter(dt)
        first_val = next(iterdt)
        num_blocks = first_val.groupby(['block_id']).mean()['icf'].shape[0]

        j=0
        icf_dict = {}
        for k in data.keys():
            if(j>=num_blocks-1):
                j=0
            grouped= data[k].groupby(['block_id']).mean()['icf']
            
            for icf in grouped:
                if(j not in icf_dict.keys()):
                    icf_dict[j] = []
                icf_dict[j].append(icf)
                j += 1
                
        for k in icf_dict.keys():
            block_avg[k] = np.mean(icf_dict[k])
            block_std[k] = np.std(icf_dict[k])
        return block_avg,block_std
    
    def icf_stats_learning(self,data):
        perf = {}
        dt = data.values()
        iterdt = iter(dt)
        first_val = next(iterdt)
        num_blocks = first_val.groupby(['block_id']).mean()['icf'].shape[0]

        for k in data.keys():
            perf[k] = []
            grouped = self.icf_per_block(data,k)
            for icf in grouped:
                perf[k].append(icf)
        y = {}
        for k in perf.keys():
            y[k]=[]
            for icf in perf[k]:
                y[k].append(np.log(icf))
        x = np.arange(num_blocks)
        st = []
        for k in y.keys():
            r = stats.linregress(x, y[k])
            st.append(r.slope)
        return st

    def hist_participant(self,data_mab,data_random,data_choi): 
        mab = len(data_mab.keys())
        random = len(data_random.keys())
        choi = len(data_choi.keys())
        max_participants = mab
        if(random > mab & random > choi):
            max_participants = random
        elif(choi > mab & choi > random):
            max_participants = choi

        fig, axes = plt.subplots(nrows=3, ncols=max_participants, figsize=(16,16))
        n = 0
        for k in data_mab.keys():
            data_mab[k]['width'].value_counts().sort_index().plot(xlabel="width",ylabel="count",title =k,kind='bar',ax=axes[0,n])
            n += 1
        n = 0
        for k in data_random.keys():
            data_random[k]['width'].value_counts().sort_index().plot(xlabel="width",ylabel="count",title =k,kind='bar',ax=axes[1,n])
            n += 1
        n = 0
        for k in data_choi.keys():
            data_choi[k]['width'].value_counts().sort_index().plot(xlabel="width",ylabel="count",title =k,kind='bar',ax=axes[2,n])
            n += 1
        fig.tight_layout()
    
    def task_distribution(self,data_mab,data_random,data_choi): 
        mab = len(data_mab.keys())
        random = len(data_random.keys())
        choi = len(data_choi.keys())
        max_participants = mab
        if(random > mab & random > choi):
            max_participants = random
        elif(choi > mab & choi > random):
            max_participants = choi

        fig, axes = plt.subplots(nrows=3, ncols=max_participants, figsize=(20,20))
        n = 0
        for k in data_mab.keys():
            axes[0,n].plot(data_mab[k]['block_id'],data_mab[k]['width'],marker='o', linestyle='dashed')
            n += 1
        n = 0
        for k in data_random.keys():
            axes[1,n].plot(data_random[k]['block_id'],data_random[k]['width'],marker='o', linestyle='dashed')
            n += 1
        n = 0
        for k in data_choi.keys():
            axes[2,n].plot(data_choi[k]['block_id'],data_choi[k]['width'],marker='o', linestyle='dashed')
            n += 1
        fig.tight_layout()

    def icf_slope(self,data,ax,a):
        n=0
        for k in data.keys():
            l = []
            grp = data[k].groupby(['block_id']).mean()['icf'].reset_index()            
            x = grp['block_id'].tolist()
            y = np.log(grp['icf'].tolist())
            r = stats.linregress(x, y)
            for i in x:
                l.append(r.slope*i +r.intercept)
            ax[a,n].plot(x, l, 'r',label = str(r.pvalue)) 
            ax[a,n].plot(x, y, 'o')
            ax[a,n].set_xlabel("block")
            ax[a,n].set_ylabel("log icf")
            ax[a,n].set_title(k)
            ax[a,n].legend()
            n += 1

    def plot_icf_slope(self,data_mab,data_random,data_choi):
        mab = len(data_mab.keys())
        random = len(data_random.keys())
        choi = len(data_choi.keys())
        max_participants = mab
        if(random > mab & random > choi):
            max_participants = random
        elif(choi > mab & choi > random):
            max_participants = choi

        fig, ax = plt.subplots(3,max_participants,figsize=(16,16))
        
        self.icf_slope(data_mab,ax,0)
        self.icf_slope(data_random,ax,1)
        self.icf_slope(data_choi,ax,2)

        fig.tight_layout()
    
    def num_participants(self,data):
        num_partcipants = []
        num_mab=0
        num_random=0
        num_choi = 0
        for k in data.keys():
            if('MAB' in k):
                num_mab += 1
            elif('Random' in k):
                num_random += 1
            else:
                num_choi += 1
        num_partcipants.append(num_mab)
        num_partcipants.append(num_random)
        num_partcipants.append(num_choi)
        return np.array(num_partcipants) 

    def plot_jerk_slope(self,data,jerk_norm):
        n = self.num_participants(data)

        fig, ax = plt.subplots(3,np.max(n),figsize=(16,16))
        
        self.jerk_slope(data,jerk_norm,ax)

        fig.tight_layout()
    
    def jerk_slope(self,data,jerk_norm,ax):
        n1=0
        n2=0
        n3=0
        jk = self.jerk_per_block(data,jerk_norm,4)
        for k in data.keys(): 
            y = np.log(jk[k])
            x = np.arange(len(y))
            r = stats.linregress(x, y)
            l=[]
            for i in x:
                l.append(r.slope*i +r.intercept)
            if('MAB' in k):
                ax[0,n1].plot(x, l, 'r',label = str(r.pvalue)) 
                ax[0,n1].plot(x, y, 'o')
                ax[0,n1].set_xlabel("block")
                ax[0,n1].set_title(k)
                ax[0,n1].legend()
                n1 += 1
            elif('Random' in k):
                ax[1,n2].plot(x, l, 'r',label = str(r.pvalue)) 
                ax[1,n2].plot(x, y, 'o')
                ax[1,n2].set_xlabel("block")
                ax[1,n2].set_title(k)
                ax[1,n2].legend()
                n2 += 1
            else:
                ax[2,n3].plot(x, l, 'r',label = str(r.pvalue)) 
                ax[2,n3].plot(x, y, 'o')
                ax[2,n3].set_xlabel("block")
                ax[2,n3].set_title(k)
                ax[2,n3].legend()
                n3 += 1
            ax[0,0].set_ylabel("log jerk")
            ax[1,0].set_ylabel("log jerk")
            ax[2,0].set_ylabel("log jerk")

    def jerk_norm(self,data,start_idx,stop_idx,num_trials):        
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
        
    def plot_jerk(self,data,jerk_data):                                               
        n = self.num_participants(data)
        fig_, ax_ = plt.subplots(nrows=3, ncols=np.max(n), figsize=(16,16))
        c = 0
        c1 = 0
        c2 = 0        

        for k in jerk_data.keys():

            if('MAB' in k):
                ax_[0,c].plot(jerk_data[k])
                ax_[0,c].set_xlabel("block")
                ax_[0,c].set_ylabel("jerk")
                ax_[0,c].set_title(k)
                c += 1
            elif('Random' in k):
                ax_[1,c1].plot(jerk_data[k])
                ax_[1,c1].set_xlabel("block")
                ax_[1,c1].set_ylabel("jerk")
                ax_[1,c1].set_title(k)
                c1 +=1
            else:
                ax_[2,c2].plot(jerk_data[k])
                ax_[2,c2].set_xlabel("block")
                ax_[2,c2].set_ylabel("jerk")
                ax_[2,c2].set_title(k)
                c2 += 1
            
        fig_.tight_layout()

    def jerk_stats_learning(self,data):  
        dt = data.values()
        iterdt = iter(dt)
        first_val = next(iterdt)
        num_trials = len(first_val)
        y = {}
        for k in data.keys():
            y[k]=[]
            for jk in data[k]:
                y[k].append(np.log(jk))
        x = np.arange(num_trials)
        st = []
        for k in y.keys():
            r = stats.linregress(x, y[k])
            st.append(r.slope)
        return st

    def jerk_stats_test(self,data):
        lst_avg_participant=[]
        for k in data.keys():
            lst_avg_participant.append(np.mean(data[k]))
        avg = np.mean(lst_avg_participant)
        std = np.std(lst_avg_participant)
        return avg,std
    
    def jerk_per_block(self,data,jerk_norm,num_trials): 
        dt = {}
        for k in data.keys():
            ct = 0
            avg_blk = []
            
            while(ct < len(jerk_norm[k])):
                slc = jerk_norm[k][ct:num_trials+ct+1]
                avg_blk.append(np.mean(slc))#jerk block avg 
                dt[k] = avg_blk
                ct += num_trials + 1
        return dt
    
    def avg_jerk_condition(self,data,jerk_norm,num_trials): 
        dt = {}
        for k in data.keys():
            ct = 0
            avg_blk = []
            
            while(ct < len(jerk_norm[k])):
                slc = jerk_norm[k][ct:num_trials+ct+1]
                avg_blk.append(np.mean(slc))#jerk block avg 
                dt[k] = avg_blk
                ct += num_trials + 1
        grp_avg_blk = {}
        for k in dt.keys():
            for i in range(len(dt[k])):
                if(i not in grp_avg_blk.keys()):
                    grp_avg_blk[i] = []
                grp_avg_blk[i].append(dt[k][i])#jerk of block k for each participant
        
        return grp_avg_blk
        
    
    def jerk_stats_transfer(self,data):
        grp_avg = {}
        grp_std = {}
        for k in data.keys():
            grp_avg[k] = np.mean(data[k])
            grp_std[k] = np.std(data[k])      
    
        return grp_avg,grp_std
    
    def prepare_df(self,data,jerk_data,condition):
        pretest_val = self.jerk_per_block(data,jerk_data,4) 
        merged_pre_lst = []
        for lst in pretest_val.values():
            merged_pre_lst += lst
        arr_con = np.full((len(merged_pre_lst)),condition)  
        dt = {"Jerk": merged_pre_lst,"Condition":arr_con}
        df = pd.DataFrame.from_dict(dt)
        return df

    def prepare_icf_df(self,test_val,condition):
        merged_pre_lst = []
        for lst in test_val.values():
            merged_pre_lst += lst
        arr_con = np.full((len(merged_pre_lst)),condition)  
        dt = {"Icf": merged_pre_lst,"Condition":arr_con}
        df = pd.DataFrame.from_dict(dt)
        return df

    def prepare_jerk_df(self,data,jerk_data,condition):
        test_val = self.jerk_per_block(data,jerk_data,4) 
        merged_pre_lst = []
        for lst in test_val.values():
            merged_pre_lst += lst
        arr_con = np.full((len(merged_pre_lst)),condition)  
        dt = {"Jerk": merged_pre_lst,"Condition":arr_con}
        df = pd.DataFrame.from_dict(dt)
        return df
    
    def prepare_icf_df(self,test_val,condition):
        merged_pre_lst = []
        for lst in test_val.values():
            merged_pre_lst += lst
        arr_con = np.full((len(merged_pre_lst)),condition)  
        dt = {"Icf": merged_pre_lst,"Condition":arr_con}
        df = pd.DataFrame.from_dict(dt)
        return df
    
    def icf_per_block_training_test(self,pretest_data,posttest_data,condition):
        df_pre_ = {}
        df_post_ = {}
        for k in posttest_data.keys():
            df_pre = pretest_data[k].groupby(['block_id']).mean()['icf'].reset_index()
            df_post = posttest_data[k].groupby(['block_id']).mean()['icf'].reset_index()
            df_pre_[k] = df_pre['icf'].tolist()
            df_post_[k] = df_post['icf'].tolist()
            
        df_pre_condition = self.prepare_icf_df(df_pre_,condition)
        df_post_condition = self.prepare_icf_df(df_post_,condition) 
        return  df_pre_condition,df_post_condition

    def icf_per_participant_training_test(self,data,condition):
        p = {}
        for k in data.keys():
            p[k] = self.avg_icf_participant(data,k)
            
        arr_con = np.full((len(p.keys())),condition) 
        dt = {"Icf":p.values() ,"Condition":arr_con}
        df = pd.DataFrame.from_dict(dt)
        return  df

    def jerk_per_participant_training_test(self,data,jerk_norm,condition):
        p = {}
        for k in data.keys(): 
            d = self.jerk_per_block(data,jerk_norm,4)
            p[k] = np.mean(d[k])
        arr_con = np.full((len(p.keys())),condition) 
        dt = {"Jerk":p.values() ,"Condition":arr_con}
        df = pd.DataFrame.from_dict(dt)
        return  df

    def slope_per_task(self,data):
        perf = {}
        grp_lst = {}
        for k in data.keys():
            df = data[k][(data[k]['width']!=75)]
            blk_df = df.groupby(['block_id','width']).mean()['icf'].reset_index()

            for b in range (len(blk_df)):
                rw = blk_df.loc[b]
                w = rw.width
                if( w not in grp_lst.keys()):
                    grp_lst[w] = []
                grp_lst[w].append(np.log(rw.icf))

            temp = grp_lst.copy()
            grp_lst.clear()
            perf[k]=temp

        lst_w = []
        lst_s = []
        for k in perf.keys():
            for i in perf[k].keys():
                x = np.arange(len(perf[k][i]))
                r = stats.linregress(x,perf[k][i])
                lst_w.append(i)
                lst_s.append(r.slope)
        return lst_w,lst_s 
    
    def icf_per_block_condition(self,data):
        df_ = {}
        merged = {}
        for k in data.keys():
            df = data[k].groupby(['block_id']).mean()['icf'].reset_index()
            icf_lst = df['icf'].tolist()
            df_[k] = icf_lst
            for i in range(len(icf_lst)):
                if (i not in merged.keys()):
                    merged[i] = []
                merged[i].append(icf_lst[i])        
        return merged


        