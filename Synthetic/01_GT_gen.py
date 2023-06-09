import numpy as np 
import random 

fn = 40
features = np.load('fusion_features.npy')
tr_num = np.load('tr_num.npy')
n_samples = len(features)
# m1 - [0:10]
# m2 - [10:20]
# m3 - [20:30]
# m4 - [30:40]

labels = np.zeros((n_samples))
exp_id = 6

for i in  range(n_samples):
    # Synthetic decision function
    F1 = np.sum(features[i,0:40])
    if F1  < -0.2:
        labels[i] = 0
    else:
        labels[i] = 1      
print( 'Prevalence is', f"{sum(labels)/n_samples:0.2f}")

grad = np.zeros((n_samples,40))
# ground truth importance using gradient of the true decision function
for i in range(len(labels)):
    grad[i,0:40] = 1 
    
imp = np.mean(np.abs(grad),0 ) # average over all test for each fold

imp_m1 = np.sum(imp[0:10])/np.sum(imp)
imp_m2 = np.sum(imp[10:20])/np.sum(imp)
imp_m3 = np.sum(imp[20:30])/np.sum(imp)
imp_m4 = np.sum(imp[30:40])/np.sum(imp)

imp_m = np.array( [ imp_m1, imp_m2, imp_m3, imp_m4] ) 

np.save('F'+str(exp_id)+'_imp_gt',imp_m)
np.save('D1_F'+str(exp_id)+'_label_train',labels[0:tr_num])
np.save('D1_F'+str(exp_id)+'_label_test',labels[tr_num:-1])
np.save('exp_id', exp_id)
