import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from mimic_models import*
import torch
import shap

def SHAP(args, background_deep_feat, deep_feat, net2, device): 
    
    net2.eval()

    # Wrap the PyTorch model with a function for use with SHAP
    def torch_model_wrapper(x):
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).to(device)
            pred = net2.forward(x_t[: , 0 : args.enc_dim], 
                                x_t[: , args.enc_dim : args.enc_dim+args.n_img_prob] ,
                                x_t[: , args.enc_dim+args.n_img_prob : args.enc_dim+args.n_img_prob+n_tab])
            
            return pred.cpu().numpy()

    # Compute SHAP values
    background_summary = shap.kmeans(background_deep_feat.cpu().numpy(), args.K)  # Choose an appropriate value for K, e.g., 100
    explainer = shap.KernelExplainer(torch_model_wrapper, background_summary)
    
    if args.type == 'G':
        imp = explainer.explain(deep_feat.detach().cpu().numpy())
    elif args.type == 'S':
        imp = explainer.shap_values(deep_feat.detach().cpu().numpy(), nsamples=200)
        imp = imp[0]
    return(imp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # File paths
    parser.add_argument("--csv_dir", default="/mnt/research/midi_lab/Muneeza_Research/MIMIC_work/", type=str)
    parser.add_argument("--out_dir", default="/mnt/research/midi_lab/Muneeza_Research/MIMIC_work/Results/", type=str) 
    # Model inputs
    parser.add_argument("--model_name", type=str, help="name of model (e.g., name of saved weights file <model_name>.pt)")
    parser.add_argument("--enc_dim", default=10 , type=int, help="dimension of encoded image")
    parser.add_argument("--n_img_prob", default=18 , type=int, help="dimension of image_probabilities")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size for training, validation, and testing (will be lowered if TTA used)")
    parser.add_argument("--dataset", default='Train' , type=str, help="Train or Test indicating which dataset to use for ")
    # SHAP inputs
    parser.add_argument("--type", default='G' , type=str, help="Global vs Sample")
    parser.add_argument("--K", default=2000 , type=int, help="number of samples to use for computing background")

    args = parser.parse_args()
    
    # Load tabular feature names
    tab_feat_names = pd.read_csv(args.csv_dir+'tab_names.csv')['0'].to_list()
    n_tab = len(tab_feat_names)

    # Load pre-trained models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_dir = os.path.join(args.out_dir, args.model_name)
    net2 = Hybrid_Fusion(n_encimg_features=args.enc_dim, n_tab_features=n_tab).to(device).float()
    net2.load_state_dict(torch.load(os.path.join(net_dir , 'net2.pt') , map_location=device) ) 
    
    # Load pre-extracted deep features 
    background_deep_feat = torch.load(os.path.join(net_dir, 'F_imp_results', 'Train_Fusion_tensor.pt')).to(device)
    deep_feat = torch.load(os.path.join(net_dir, 'F_imp_results', f'{args.dataset}_Fusion_tensor.pt')).to(device)

    imp_s = SHAP(args, background_deep_feat, deep_feat, net2, device)

    if args.type == 'G':
        imp = np.abs(imp_s)[:,0]

    elif args.type =='S':
        imp = np.mean((np.abs(imp_s)),0)

    df = pd.read_csv(os.path.join(net_dir, 'input_fusion_indexing.csv'),index_col=0)
    input_list = df.index.values

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
    ax.grid(alpha=0.4)

    xlabel = input_list
    imp_in_plot , ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.bar(np.arange(len(xlabel)), imp_in,  edgecolor='k', width=0.25, capsize=2,  label='Predicted')
    ax.set_xticks(np.arange(len(xlabel)))
    ax.set_xticklabels(xlabel, fontsize=12)
    ax.set_ylabel('Normalized importance',fontsize=12)
    ax.set_ylim([0,1])
    ax.legend()
    ax.grid(alpha=0.4)

    out_dir = os.path.join(net_dir, "F_imp_results")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    imp_in_plot.savefig(os.path.join(out_dir, f"{args.dataset}_{args.type}SHAP_in_imp.pdf"), bbox_inches="tight")
    imp_m_plot.savefig(os.path.join(out_dir, f"{args.dataset}_{args.type}SHAP_m_imp.pdf"), bbox_inches="tight")

    if args.type =='S':
        np.save(os.path.join(out_dir, f"{args.dataset}_{args.type}SHAP_imp_sample"), imp_s)
        
    np.save(os.path.join(out_dir, f"{args.dataset}_{args.type}SHAP_imp_raw"), imp)
    np.save(os.path.join(out_dir, f"{args.dataset}_{args.type}SHAP_imp_modality_norm"), imp_m)
    np.save(os.path.join(out_dir, f"{args.dataset}_{args.type}SHAP_imp_input_norm"), imp_in)