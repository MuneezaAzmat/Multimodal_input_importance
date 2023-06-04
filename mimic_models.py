import torch
import torchxrayvision as xrv


class DenseNetFeatureExtractor(torch.nn.Module):
    def __init__(self, n_encimg_features, weights, frozen):
        super(DenseNetFeatureExtractor, self).__init__()
        
        self.weights = weights

        if self.weights is not None:
            self.Multi_DenseNet = xrv.models.DenseNet(weights=weights) 
        else:
            self.Multi_DenseNet = xrv.models.DenseNet()

        self.num_features = self.Multi_DenseNet.classifier.in_features

        if frozen:
            for name, param in self.Multi_DenseNet.named_parameters():
                if name not in ['classifier.weight', 'classifier.bias']:
                    param.requires_grad = False

        layers_keep = list(self.Multi_DenseNet.children())[:-2]
        
        self.binary_DenseNet = torch.nn.Sequential(*layers_keep)
        self.relu = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(self.num_features, n_encimg_features)
        self.batch_norm = torch.nn.BatchNorm1d(n_encimg_features)

    def forward(self, x):
        x_prob = self.Multi_DenseNet(x)
        x = self.binary_DenseNet(x)         # out dim: [batch, 1024, 7, 7]
        x = self.relu(x)                    # out dim: [batch, 1024, 7, 7]
        x = self.avgpool(x)                 # out dim: [batch, 1024, 1, 1]
        x = self.flatten(x)                 # out dim: [batch, 1024]
        x = self.linear(x)                  # out dim: [batch, n_enc_img_features]    
        x = self.batch_norm(x)              # out dim: [batch, n_enc_img_features]
        return x , x_prob


class Hybrid_Fusion(torch.nn.Module):
    def __init__(self, n_encimg_features, n_tab_features):
        super(Hybrid_Fusion, self).__init__()

        self.n_tab_features = n_tab_features
        self.n_encimg_features = n_encimg_features
        self.n_img_prob = 18

        self.fused_nn = torch.nn.Sequential(
            torch.nn.Linear(self.n_encimg_features + self.n_img_prob+ self.n_tab_features, 100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(50, 1),
            torch.nn.Sigmoid()
        )

        # Needed only if enc_img and tabular features are not normalized
        # self.bn = torch.nn.BatchNorm1d(self.n_encimg_features + self.n_tab_features)

    def forward(self, enc_im, im_prob, tabular):
        x1 = enc_im
        x2 = im_prob
        x3 = tabular

        x = torch.cat((x1, x2, x3), dim=-1)
        x = self.fused_nn.forward(x)
        return x
    
