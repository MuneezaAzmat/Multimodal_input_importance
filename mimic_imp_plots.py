import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from scipy.stats import pearsonr
# import rbo

def krippendorff_alpha_continuous(data):
    """
    Calculates Krippendorff's alpha for continuous variables.
    Args:
        data (array-like): A list or array of data with shape (n_raters, n_items).
    Returns:
        alpha (float): Krippendorff's alpha for continuous variables.
    """
    # Flatten the data into a 1D array
    data_flat = np.ravel(data)
    n = len(data_flat)
    
    # Calculate the mean of each item across all raters
    item_means = np.mean(data, axis=0)
    
    # Calculate the squared differences between each rating and the item mean
    sq_diffs = np.square(data - item_means)
    
    # Calculate the sum of squared differences across all ratings
    S = np.sum(sq_diffs)
    
    # Calculate the Pearson correlation coefficient between all pairs of raters
    corrs = np.corrcoef(data, rowvar=False)
    np.fill_diagonal(corrs, 0)
    r = np.mean(corrs)
    
    # Calculate Krippendorff's alpha
    alpha = 1 - (S / (n * (n - 1))) / (np.square(np.std(data_flat)) - np.square(np.mean(np.std(data, axis=1)))) * (1 - r)
    
    return alpha

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Ariel']
fsz=14
total_w = 0.7
n_bars = 5
bar_w = total_w / n_bars
xlabel = ['Image', 'Age' , 'Gender','Insurance', 'Marital_st', 'Ethnicity']
x = np.arange(len(xlabel))
y = [0,0.2,0.4,0.6,0.8,1.0]
x_offset = (np.arange(n_bars) - n_bars / 2) * bar_w + bar_w / 2

font = FontProperties(size=fsz)

load_dir = '/mnt/research/midi_lab/Muneeza_Research/MIMIC_work/Results/2023-06-01_encdim10_Cardiomegaly_im_freeze0/F_imp_results/'
dataset = 'Train'
imp_grad = np.load(load_dir+f'{dataset}_GRAD_imp_input_norm.npy')
imp_perm = np.load(load_dir+f'{dataset}_PERM_imp_input_norm.npy')
imp_lime = np.load(load_dir+f'{dataset}_LIME_imp_input_norm.npy')
imp_shap = np.load(load_dir+f'{dataset}_GSHAP_imp_input_norm.npy')

imp_agg = np.mean((imp_grad, imp_lime, imp_perm, imp_shap),0)

np.save(os.path.join(load_dir, f"{dataset}_AGG_imp_input_norm"), imp_agg)

imp_plot , ax = plt.subplots(1, 1, figsize=(5, 3.5))

ax.bar(x+x_offset[0] ,imp_grad,  edgecolor='k',width=bar_w, capsize=2, color='C1', label='GRAD')
ax.bar(x+x_offset[1] ,imp_perm,  edgecolor='k',width=bar_w, capsize=2, color='C2', label='PERM')
ax.bar(x+x_offset[2] ,imp_lime,  edgecolor='k',width=bar_w, capsize=2, color='C3', label='LIME')
ax.bar(x+x_offset[3] ,imp_shap,  edgecolor='k',width=bar_w, capsize=2, color='C4', label='SHAP')
ax.bar(x+x_offset[4] ,imp_agg,   edgecolor='k',width=bar_w, capsize=2, color='C0', label='AGG')

ax.set_xticks(x)
ax.set_xticklabels(xlabel, rotation= 70 , fontproperties=font)
ax.set_ylim([0,1])
ax.grid(alpha=0.3)
ax.set_yticks(y)
ax.set_yticklabels([str(i) for i in y], fontproperties=font)
ax.set_ylabel('Normalized Importance',fontproperties=font)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
imp_plot.savefig(load_dir+'Cardiomeg_input_imp_mm.png', dpi = 300, bbox_inches="tight")

imp_agg_plot , ax = plt.subplots(1, 1, figsize=(5, 3.5))
ax.bar(x ,imp_agg,  edgecolor='k',width=bar_w, capsize=2)
ax.set_xticks(x)
ax.set_xticklabels(xlabel, rotation= 70 , fontproperties=font)
ax.set_ylim([0,1])
ax.grid(alpha=0.3)
ax.set_yticks(y)
ax.set_yticklabels([str(i) for i in y], fontproperties=font)
ax.set_ylabel('Normalized Aggregated Importance',fontproperties=font)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
imp_agg_plot.savefig(load_dir+'Cardiomeg_imp_agg.png', dpi = 300, bbox_inches="tight")

rank_shap = np.argsort(np.argsort(-imp_shap))
rank_grad = np.argsort(np.argsort(-imp_grad))
rank_lime = np.argsort(np.argsort(-imp_lime))
rank_perm = np.argsort(np.argsort(-imp_perm))

# print(rank_grad, rank_lime, rank_perm)

# table_1 = pd.DataFrame(columns = ["", "SHAP",  "LIME", "GRAD", "PERM"])
# p = 0.99

# table_1.loc[0]= ["SHAP" , rbo.RankingSimilarity(rank_shap,rank_shap).rbo() , rbo.RankingSimilarity(rank_shap,rank_lime).rbo(), 
#                  rbo.RankingSimilarity(rank_shap,rank_grad).rbo(), rbo.RankingSimilarity(rank_shap,rank_perm).rbo()]

# table_1.loc[1]= ["LIME" , rbo.RankingSimilarity(rank_lime,rank_shap).rbo() , rbo.RankingSimilarity(rank_lime,rank_lime).rbo(),
#                   rbo.RankingSimilarity(rank_lime,rank_grad).rbo(),  rbo.RankingSimilarity(rank_lime,rank_perm).rbo()]

# table_1.loc[2]= ["GRAD" , rbo.RankingSimilarity(rank_grad,rank_shap).rbo() , rbo.RankingSimilarity(rank_grad,rank_lime).rbo(), 
#                  rbo.RankingSimilarity(rank_grad,rank_grad).rbo(), rbo.RankingSimilarity(rank_grad,rank_perm).rbo()]

# table_1.loc[3]= ["PERM" , rbo.RankingSimilarity(rank_perm,rank_shap).rbo() , rbo.RankingSimilarity(rank_perm,rank_lime).rbo(), 
#                  rbo.RankingSimilarity(rank_perm,rank_grad).rbo(), rbo.RankingSimilarity(rank_perm,rank_perm).rbo()]

# print("---------------------------------------------------")
# print(table_1.round(2).to_latex(index=False))


def diff(x,y):
    d = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
    # d = np.sum(np.abs(x-y))
    # d=d*100
    # d = np.linalg.norm(x-y)
    return (d)

table_2 = pd.DataFrame(columns = ["", "SHAP",  "LIME", "GRAD", "PERM"])
table_2.loc[0]= ["SHAP" , diff(imp_shap,imp_shap) , diff(imp_shap,imp_lime), 
                 diff(imp_shap,imp_grad), diff(imp_shap,imp_perm) ]

table_2.loc[1]= ["LIME" , diff(imp_lime,imp_shap) , diff(imp_lime,imp_lime), 
                 diff(imp_lime,imp_grad), diff(imp_lime,imp_perm)]

table_2.loc[2]= ["GRAD" , diff(imp_grad,imp_shap) , diff(imp_grad,imp_lime), 
                 diff(imp_grad,imp_grad), diff(imp_grad,imp_perm)]

table_2.loc[3]= ["PERM" , diff(imp_perm,imp_shap) , diff(imp_perm,imp_lime), 
                 diff(imp_perm,imp_grad), diff(imp_perm,imp_perm)]


print("---------------------------------------------------")
print('Cosine similarity')
print(table_2.round(2).to_latex(index=False))

from scipy.stats import spearmanr

def diff_m(x, y):
    rank_x = np.argsort(np.argsort(x))[::-1]
    rank_y = np.argsort(np.argsort(y))[::-1]
    corr_coef, _ = spearmanr(rank_x, rank_y)
    return corr_coef

table_3 = pd.DataFrame(columns = ["", "SHAP",  "LIME", "GRAD", "PERM"])
table_3.loc[0]= ["SHAP" , diff_m(imp_shap,imp_shap) , diff_m(imp_shap,imp_lime), 
                 diff_m(imp_shap,imp_grad), diff_m(imp_shap,imp_perm) ]

table_3.loc[1]= ["LIME" , diff_m(imp_lime,imp_shap) , diff_m(imp_lime,imp_lime), 
                 diff_m(imp_lime,imp_grad), diff_m(imp_lime,imp_perm)]

table_3.loc[2]= ["GRAD" , diff_m(imp_grad,imp_shap) , diff_m(imp_grad,imp_lime), 
                 diff_m(imp_grad,imp_grad), diff_m(imp_grad,imp_perm)]

table_3.loc[3]= ["PERM" , diff_m(imp_perm,imp_shap) , diff_m(imp_perm,imp_lime), 
                 diff_m(imp_perm,imp_grad), diff_m(imp_perm,imp_perm)]

print("---------------------------------------------------")
print('Spearman for ranking')
print(table_3.round(2).to_latex(index=False))
