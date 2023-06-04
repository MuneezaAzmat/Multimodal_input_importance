import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from mimic_models import *
from mimic_dataset import *
from torch.utils.data import DataLoader


def gradient(args, tab_feat_names, n_tab):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = PathologyDataset(os.path.join(args.csv_dir, f'{args.pathology}_Train_{args.train_type}.csv'), img_dir=args.img_dir ,  tab_feat_names=tab_feat_names, pathology=args.pathology)
    # batch size 1 for gradient 
    data_loader  = DataLoader(data, shuffle=False, batch_size=1, num_workers=4)
    fn = args.enc_dim + args.n_img_prob + n_tab

    net_dir = os.path.join(args.out_dir, args.model_name)
    net1 = DenseNetFeatureExtractor(n_encimg_features= args.enc_dim, weights="densenet121-res224-chex", frozen=1).to(device).float()
    net2 = Hybrid_Fusion(n_encimg_features=args.enc_dim, n_tab_features=len(tab_feat_names)).to(device).float()

    net1.load_state_dict(torch.load(os.path.join(net_dir , 'net1.pt') , map_location=device) )
    net2.load_state_dict(torch.load(os.path.join(net_dir , 'net2.pt') , map_location=device) ) 

    net1.eval()
    net2.eval()

    grad = np.zeros((len(data), fn))

    for i, (image, tabular, label) in enumerate(data_loader, 0):
        img = image.to(device)
        tab = tabular.to(device)

        enc_img , img_prob = net1.forward(img)

        # enc_img.requires_grad = True
        # img_prob.requires_grad = True
        tab.requires_grad = True
        
        y_prob = net2.forward(enc_img, img_prob, tab)
        
        grad[i, 0:args.enc_dim] = torch.autograd.grad(y_prob, enc_img, retain_graph=True)[0].cpu().numpy()
        grad[i, args.enc_dim: args.enc_dim+args.n_img_prob] = torch.autograd.grad(y_prob, img_prob, retain_graph=True)[0].cpu().numpy()
        grad[i, args.enc_dim+args.n_img_prob : args.enc_dim+args.n_img_prob+n_tab] = torch.autograd.grad(y_prob, tab, retain_graph=True)[0].cpu().numpy()

    imp = np.mean(np.abs(grad),0 )
    imp_sample = np.abs(grad)
    return (imp, imp_sample)    

if __name__ == "__main__":
   # Command-line arguments
    parser = argparse.ArgumentParser()
    # File paths
    parser.add_argument("--csv_dir", default="/mnt/research/midi_lab/Muneeza_Research/MIMIC_work/", type=str)
    parser.add_argument("--img_dir", default="/mnt/research/midi_lab/Muneeza_Research/MIMIC_work/Preprocessed_img/", type=str)
    parser.add_argument("--out_dir", default="/mnt/research/midi_lab/Muneeza_Research/MIMIC_work/Results/", type=str) 
    parser.add_argument("--train_type", default='im', type=str, help="imbalance, upsample, downsample")
    # Model inputs
    parser.add_argument("--model_name", type=str, help="name of model (e.g., name of saved weights file <model_name>.pt)")
    parser.add_argument("--enc_dim", default=10 , type=int, help="dimension of encoded image")
    parser.add_argument("--n_img_prob", default=18 , type=int, help="dimension of image_probabilities")
    parser.add_argument("--dataset", default='Train' , type=str, help="Train or Test indicating which dataset to use for ")
    parser.add_argument("--pathology", default='Cardiomegaly', type=str, help="classification task")
    
    args = parser.parse_args()

    tab_feat_names = pd.read_csv(args.csv_dir+'tab_names.csv')['0'].to_list()
    n_tab = len(tab_feat_names)

    imp , imp_sample = gradient(args, tab_feat_names= tab_feat_names, n_tab=n_tab)
    print('raw', imp)

    df = pd.read_csv(os.path.join(args.out_dir, args.model_name, 'input_fusion_indexing.csv'), index_col=0)
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
    ax.grid(alpha=0.2)

    xlabel = ['Image', 'Age' , 'Gender','Insurance', 'Marital_st', 'Ethnicity']
    imp_in_plot , ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.bar(np.arange(len(xlabel)), imp_in,  edgecolor='k', width=0.25, capsize=2,  label='Predicted')
    ax.set_xticks(np.arange(len(xlabel)))
    ax.set_xticklabels(xlabel, fontsize=12)
    ax.set_ylabel('Normalized importance',fontsize=12)
    ax.set_ylim([0,1])
    ax.legend()
    ax.grid(alpha=0.2)

    out_dir = os.path.join(args.out_dir, args.model_name, "F_imp_results")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    imp_in_plot.savefig(os.path.join(out_dir, f"{args.dataset}_GRAD_in_imp.pdf"), bbox_inches="tight")
    imp_m_plot.savefig(os.path.join(out_dir, f"{args.dataset}_GRAD_m_imp.pdf"), bbox_inches="tight")
    np.save(os.path.join(out_dir, f"{args.dataset}_GRAD_imp_raw"), imp)
    np.save(os.path.join(out_dir, f"{args.dataset}_GRAD_imp_modality_norm"), imp_m)
    np.save(os.path.join(out_dir, f"{args.dataset}_GRAD_imp_input_norm"), imp_in)
    # np.save(os.path.join(out_dir, f"{args.dataset}_GRAD_imp_sample"), imp_sample)