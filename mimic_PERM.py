import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from mimic_models import *
from mimic_dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

def balanced_acc (y_true, y_pred ):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    if tp + fn > 0:
        re = tp / (tp + fn)
    else:
        re = np.nan
    if tn + fp > 0:
        sp = tn / (tn + fp)
    else:
        sp = np.nan
    balance_accuracy = (re + sp) / 2
    return(balance_accuracy)


def gen_deep_feat(args, net1, net2, device, tab_feat_names, n_tab):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = PathologyDataset(os.path.join(args.csv_dir, f'{args.pathology}_Train_{args.train_type}.csv'), img_dir=args.img_dir ,  tab_feat_names=tab_feat_names, pathology=args.pathology)
    data_loader  = DataLoader(data, shuffle=False, batch_size=args.batch_size, num_workers=4)

    net1.eval()
    net2.eval()

    ## Save Deep features at fusion layer (needed for perm, LIME, SHAP)
    sz = len(data_loader)*args.batch_size
    deep_feat = torch.zeros(sz , args.enc_dim + args.n_img_prob + n_tab)
    labels = torch.zeros(sz,1)

    with torch.no_grad():
        for i, (image, tabular, label) in enumerate(data_loader, 0):
            img = image.to(device)
            tab = tabular.to(device)
            y = label.to(device)

            enc_img , img_prob = net1.forward(img)
            len_y = len(y)
            deep_feat[i*args.batch_size:i*args.batch_size+len_y] = torch.cat((enc_img, img_prob, tab), dim=-1)
            # Save the true labels corresponding all deep features
            labels[i*args.batch_size:i*args.batch_size+len_y,:] = y
    
    labels = labels[0:i*args.batch_size+len_y,:]        
    deep_feat = deep_feat[0:i*args.batch_size+len_y, :]

    save_dir = os.path.join(net_dir, 'F_imp_results')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    torch.save(deep_feat, os.path.join(save_dir,f'{args.dataset}_Fusion_tensor.pt')) 
    torch.save(labels, os.path.join(save_dir,f'{args.dataset}_Labels_Fusion_tensor.pt')) 
    
    return(deep_feat, labels)


def permutation(args,deep_feat,y, net1, net2, device, tab_feat_names, n_tab):
    # Function to test the model with the test dataset and print the accuracy for the test images
    net1.eval()
    net2.eval()

    fn = deep_feat.shape[1]
    sz = len(deep_feat)
    imp = np.zeros(fn)

    with torch.no_grad():
        y = y.to(device)
        deep_feat = deep_feat.to(device)
        
        y_prob = net2.forward(deep_feat[: , 0 : args.enc_dim], 
                              deep_feat[: , args.enc_dim : args.enc_dim+args.n_img_prob] ,
                              deep_feat[: , args.enc_dim+args.n_img_prob : args.enc_dim+args.n_img_prob+n_tab])
        
        thr = args.thr
        y_pred = (y_prob >= thr)
        y_pred = y_pred.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        # base_accuracy = balanced_acc(y, y_pred)
        base_accuracy = sum((y_pred == y))/sz

        print('reference_accuracy', base_accuracy)

        k=0
        while k < fn:
            accuracy = 0.0
            for i in range(args.perm_iter):
                idx = torch.randperm(sz)
                perm_deep_feat = deep_feat.clone()

                perm_deep_feat[:,k] = deep_feat[idx,k]
                perm_deep_feat = perm_deep_feat.to(device)

                y_prob = net2.forward(perm_deep_feat[: , 0 : args.enc_dim], 
                                      perm_deep_feat[: , args.enc_dim : args.enc_dim+args.n_img_prob],
                                      perm_deep_feat[: , args.enc_dim+args.n_img_prob : args.enc_dim+args.n_img_prob+n_tab]) 
                
                y_pred = (y_prob >= thr)
                y_pred = y_pred.cpu().detach().numpy()
               
                # accuracy += balanced_acc(y,y_pred)
                accuracy += sum((y_pred == y))/sz

            # compute the accuracy over all test images
            accuracy = (accuracy / args.perm_iter)
            imp[k] = base_accuracy - accuracy 
            k+=1

    return(imp)

if __name__ == "__main__":
    # Parse indication argument
    parser = argparse.ArgumentParser()
    # File paths
    parser.add_argument("--csv_dir", default="/mnt/research/midi_lab/Muneeza_Research/MIMIC_work/", type=str)
    parser.add_argument("--img_dir", default="/mnt/research/midi_lab/Muneeza_Research/MIMIC_work/Preprocessed_img/", type=str)
    parser.add_argument("--out_dir", default="/mnt/research/midi_lab/Muneeza_Research/MIMIC_work/Results/", type=str) 
    # Model inputs
    parser.add_argument("--model_name", type=str, help="name of model (e.g., name of saved weights file <model_name>.pt)")
    parser.add_argument("--enc_dim", default=10 , type=int, help="dimension of encoded image")
    parser.add_argument("--n_img_prob", default=18 , type=int, help="dimension of image_probabilities")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size for training, validation, and testing (will be lowered if TTA used)")
    parser.add_argument("--dataset", default='Train' , type=str, help="Train or Test indicating which dataset to use for ")
    parser.add_argument("--train_type", default='im', type=str, help="imbalance, upsample, downsample")
    parser.add_argument("--pathology", default='Cardiomegaly', type=str, help="classification task")
    # Permutation inputs
    parser.add_argument("--perm_iter", default=500, type=int, help="number of permutations for PERM imp")  
    parser.add_argument("--gen_deep", default='True', type=str, help='compute encoded deep features again True/False ')
    parser.add_argument("--thr", default=0.5 , type=float, help="threshold for classification")
    parser.add_argument("--gen_fusion_index", default='True', type=str, help='compute index for inputs in the fusion layer True/False ')
    
    args = parser.parse_args()

    # Load tabular feature names
    tab_feat_names = pd.read_csv(args.csv_dir+'tab_names.csv')['0'].to_list()
    n_tab = len(tab_feat_names)

    # Load pre-trained models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_dir = os.path.join(args.out_dir, args.model_name)
    net1 = DenseNetFeatureExtractor(n_encimg_features= args.enc_dim, weights="densenet121-res224-chex", frozen=1).to(device).float()
    net2 = Hybrid_Fusion(n_encimg_features=args.enc_dim, n_tab_features=len(tab_feat_names)).to(device).float()
    net1.load_state_dict(torch.load(os.path.join(net_dir , 'net1.pt') , map_location=device) )
    net2.load_state_dict(torch.load(os.path.join(net_dir , 'net2.pt') , map_location=device) ) 

    if bool(args.gen_deep):
        deep_feat, labels = gen_deep_feat(args , net1, net2, device, tab_feat_names, n_tab)
    else: 
        deep_feat = torch.load(os.path.join(net_dir, 'F_imp_results', f'{args.dataset}_Fusion_tensor.pt')).to(device)
        labels = torch.load(os.path.join(net_dir, 'F_imp_results', f'{args.dataset}_Labels_Fusion_tensor.pt')).to(device)
    
    imp = permutation(args, deep_feat, labels, net1, net2, device, tab_feat_names, n_tab)

    print('raw deep feat importance: ', imp)
    input_list = ['Image','Age' , 'Gender','Insurance', 'Marital_st', 'Ethnicity']

    if bool(args.gen_fusion_index):
        df = pd.DataFrame(columns=['dim_in_fusion' , 'st_id', 'en_id'])
        df.dim_in_fusion = [args.enc_dim+args.n_img_prob , 1 , 2, 3, 4, 6 ]
        df.index = input_list
        ids = [0]
        for i in range(1, len(df)):
            ids.append(ids[i-1]+df.dim_in_fusion[i-1])
        ids.append(sum(df.dim_in_fusion))
        df.st_id = ids[0:len(df)]
        df.en_id = ids[1:len(df)+1] 
        df.to_csv(os.path.join(net_dir,'input_fusion_indexing.csv'))
    else: 
        df = pd.read_csv(os.path.join(net_dir, 'input_fusion_indexing.csv'),index_col=0)

    # MODALITY LEVEL
    imp_m = np.zeros(2)
    imp_m[0] = np.sum(imp[0 : args.enc_dim+args.n_img_prob])                         
    imp_m[1] = np.sum(imp[args.enc_dim+args.n_img_prob : args.enc_dim+args.n_img_prob+n_tab])
    imp_m = imp_m/sum(imp_m)
    imp_m[imp_m < 0] = 0

    # INPUT LEVEL
    imp_in = np.zeros(len(input_list))
    for i in range(len(input_list)):
        imp_in[i] = np.sum(imp[df.loc[input_list[i],'st_id'] : df.loc[input_list[i],'en_id']])
    imp_in = imp_in/sum(imp_in)
    imp_in[imp_in < 0] = 0

    print('normalized input importances',np.around(imp_in,3))

    xlabel = ['Image', 'Tabular']
    imp_m_plot , ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.bar(np.arange(len(xlabel)), imp_m,  edgecolor='k', width=0.25, capsize=2,  label='Predicted')
    ax.set_xticks(np.arange(len(xlabel)))
    ax.set_xticklabels(xlabel, fontsize=12)
    ax.set_ylabel('Normalized importance',fontsize=12)
    ax.set_ylim([0,1])
    ax.legend()
    ax.grid(alpha=0.2)

    xlabel = input_list
    imp_in_plot , ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.bar(np.arange(len(xlabel)), imp_in,  edgecolor='k', width=0.25, capsize=2,  label='Predicted')
    ax.set_xticks(np.arange(len(xlabel)))
    ax.set_xticklabels(xlabel, fontsize=12)
    ax.set_ylabel('Normalized importance',fontsize=12)
    ax.set_ylim([0,1])
    ax.legend()
    ax.grid(alpha=0.2)

    out_dir = os.path.join(net_dir, "F_imp_results")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    imp_in_plot.savefig(os.path.join(out_dir, f"{args.dataset}_PERM_in_imp.pdf"), bbox_inches="tight")
    imp_m_plot.savefig(os.path.join(out_dir, f"{args.dataset}_PERM_m_imp.pdf"), bbox_inches="tight")
    np.save(os.path.join(out_dir, f"{args.dataset}_PERM_imp_raw"), imp)
    np.save(os.path.join(out_dir, f"{args.dataset}_PERM_imp_modality_norm"), imp_m)
    np.save(os.path.join(out_dir, f"{args.dataset}_PERM_imp_input_norm"), imp_in)





