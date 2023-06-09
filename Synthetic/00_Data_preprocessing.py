import matplotlib.pyplot as plt 
import numpy as np 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

def pre_proc (filename):
    # Transforms images to a PyTorch Tensor
    tensor_transform = transforms.ToTensor()
  
    # Download the MNIST Dataset
    data_zip = np.load(filename)
    dataset = torch.from_numpy(np.concatenate((data_zip['train_images'],data_zip['val_images'],
                          data_zip['test_images']),0).astype(float)) 

    means = dataset.mean([1,2])
    stds  = dataset.std([1,2])
    mask = (stds!=0)

    stds = stds[mask]
    means = means[mask]
    dataset = dataset[mask]

    norm_tr = transforms.Compose([ transforms.Normalize(mean=means, std=stds, inplace=False) ])
    dataset = norm_tr(dataset)
    dataset = dataset.unsqueeze(1)
    return(dataset)
  
 #X1 and X2 
# Download the MNIST Dataset
dataset1 = pre_proc('organamnist.npz')
dataset2 = pre_proc('organcmnist.npz')

# Same images from each dataset 
ld2 = len(dataset2)
dataset1 = dataset1[0:ld2,:,:,:]
dataset2 = dataset2[:,:,:,:]

# Data Loaders
loader1 = torch.utils.data.DataLoader(dataset = dataset1, batch_size = 32, shuffle = True)
loader2 = torch.utils.data.DataLoader(dataset = dataset2, batch_size = 32, shuffle = True)

class Encoder(nn.Module):
    def __init__(self, enc_dim):
        super().__init__()
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.flatten = nn.Flatten()
        
        self.encoder_lin = nn.Sequential(
            nn.Linear(64* 5* 5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, enc_dim)
            )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, enc_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(enc_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 5 * 5)
            )
        
        self.unflatten = nn.Unflatten(1, (64, 5, 5))
        
        self.decoder_cnn = nn.Sequential(
            
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 16, 9),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 1, 12),  
            nn.BatchNorm2d(1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_cnn(x)
        x = torch.sigmoid(x)
        return x

### Define the loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.001

### Set the random seed for reproducible results
# torch.manual_seed(0)

# Model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(10)
decoder = Decoder(10)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)

def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train().double()
    decoder.train().double()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
#         print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)
  
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data
  
# Save model 
torch.save(encoder, './encoder.pt')
torch.save(decoder, './decoder.pt')
# encoder = torch.load('./encoder.pt')
# decoder = torch.load('./decoder.pt')

# Image feature extractor
# Extract deep features using trained encoder
d = 10
img_feat1 = np.zeros((ld2,d))
img_feat2 = np.zeros((ld2,d))


for i in range(ld2):
    img1 = dataset1[i].unsqueeze(0).to(device)
    img2 = dataset2[i].unsqueeze(0).to(device)
    encoder.eval()
    with torch.no_grad():
        enc1  = encoder(img1)
        enc2  = encoder(img2)
        img_feat1[i] = enc1.cpu().squeeze().numpy()
        
        
# Sample clinical data
# AMA Mods to save standard scaler
scaler1 = StandardScaler()
scaler2 = StandardScaler()

n_samples = img_feat1.shape[0]; #len(im_deep)
n_tab = 20
covar_1 = np.eye(n_tab)
covar_1[2:5,8] = 0.5
covar_1[8,2:5] = 0.5

m1 = img_feat1[0:n_samples,:]
m1 = scaler1.fit_transform(m1)

m2 = img_feat2[0:n_samples,:]
scaler2.fit(m2)
m2 = scaler2.transform(m2)

# sample both M3 and M4 together
m3 = np.random.multivariate_normal(np.zeros(n_tab), covar_1, size=n_samples)
labels = np.zeros((n_samples))

# deep features
features = np.concatenate((m1,m2,m3),1)

# input data to the multimodal model
multi_data = np.concatenate((dataset1[0:n_samples,:,:,:].squeeze().reshape(m1.shape[0],28*28).cpu().numpy(),
                             dataset2[0:n_samples,:,:,:].squeeze().reshape(m2.shape[0],28*28).cpu().numpy(),m3),1)
np.save('fusion_features', features)
np.save('inp_multi_data', multi_data)

tr_num =  int(len(labels) - len(labels)*0.3)

np.save('D1_train.npy',multi_data[0:tr_num,:])
np.save('D1_fusion_train.npy', features[0:tr_num,:] )

np.save('D1_test.npy',multi_data[tr_num:-1,:])
np.save('D1_fusion_test.npy', features[tr_num:-1,:] )

np.save('tr_num', tr_num)
