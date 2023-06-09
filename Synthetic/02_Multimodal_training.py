import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

exp_id = np.load('exp_id.npy')
## Data loader

class CustomDataset():
    def __init__(self, fname_d, fname_l, transforms=True):
        self.data = np.load(fname_d)
        self.labels = np.load(fname_l)
        self.transforms = transforms

    def __getitem__(self, index):
        label = np.array(self.labels[index])
        m1 = self.data[index,0:784].reshape(1,28,28).astype(float)
        m2 = self.data[index,784:1568].reshape(1,28,28).astype(float)
        m3 = self.data[index,1568:1568+20].reshape(20).astype(float)
        # Transform to tensor
        if self.transforms:
            m1_as_tensor = torch.from_numpy(m1)
            m2_as_tensor = torch.from_numpy(m2)
            m3_as_tensor = torch.from_numpy(m3)
            label_as_tensor = torch.from_numpy(label).type(torch.long)
            
        # Return image and the label
        return (m1_as_tensor, m2_as_tensor, m3_as_tensor, label_as_tensor)

    def __len__(self):
        return len(self.data)
        

## Model architecture

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
        
        self.norm = nn.BatchNorm1d(enc_dim)
        
    def forward(self, m1):
        x = self.encoder_cnn(m1)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = self.norm(x)
        return x
    

class Fused_net(nn.Module):
    def __init__(self):
        super(Fused_net, self).__init__()
       
        self.fused_nn = torch.nn.Sequential(
            nn.Linear(10+10+10+10, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2) ) 
        
        self.norm = nn.BatchNorm1d(40)
                    
    def forward(self, x1, x2, m3):
        
        x = torch.cat((x1, x2, m3), dim=1)
#         x = self.norm(x)
        x = self.fused_nn(x)
        
        output = F.log_softmax(x, dim=1)
        return output
# Function to test the model with the test dataset and print the accuracy for the test images
def Accuracy(device, loader):
    net1.eval()
    net2.eval()
    
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in loader:
            m1, m2, m3, labels = data
            img1 = m1.to(device)
            img2 = m2.to(device)
            tab = m3.to(device)
            labels = labels.to(device)
            
            # run the model on the test set to predict labels
            enc_img1 = net1(img1)
            enc_img2 = net1(img2)
            outputs = net2(enc_img1, enc_img2, tab)
            
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, -1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)
# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(net1, net2 , train_loader, loss, optimizer, num_epochs, device):
    best_accuracy = 0.0
    losses =[]
    
    # Define your execution device
    # Convert model parameters and buffers to CPU or Cuda
    net1.to(device).train()
    net2.to(device).train()
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        start_time = time.time()
        for i, (m1, m2, m3, labels) in enumerate(train_loader, 0):
            # get the inputs
            img1 = Variable(m1.to(device))
            img2 = Variable(m2.to(device))
            tab = Variable(m3.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # run the model on the test set to predict labels
            enc_img1 = net1(img1)
            enc_img2 = net1(img2)
            outputs = net2(enc_img1, enc_img2, tab)
            
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            
            # backpropagate the loss
            loss.backward()
            
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            losses.append(running_loss)
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                
                running_loss = 0.0
                
        val_acc = Accuracy(device, test_loader)
        print('Epoch', epoch+1, 'Train acc %d %%' % (Accuracy(device, train_loader)) , 'Val acc %d %%' % (val_acc) , "Time elapsed: ", time.time() - start_time )
        if val_acc > 95:
            break
            
    return (losses)
      
if __name__ == '__main__':
    
    batch_size = 128
    epochs = 20
    custom_train_data = CustomDataset('D1_train.npy', 'D1_F'+str(exp_id)+'_label_train.npy')
    custom_test_data = CustomDataset('D1_test.npy', 'D1_F'+str(exp_id)+'_label_test.npy')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net1 = Encoder(10).double()
    net2 = Fused_net().double()
    
    train_loader = DataLoader(dataset=custom_train_data, shuffle=True, batch_size= batch_size)
    test_loader   = DataLoader(dataset=custom_test_data, shuffle=False, batch_size= batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net1.parameters(), lr=0.001, weight_decay=0.0001)

    optimizer.add_param_group(torch.optim.Adam(net2.parameters()).param_groups[0])
    
    loss = train(net1, net2, train_loader, loss_fn, optimizer, epochs , device)

    print('Finished Training')
## Save trained model 
torch.save(net1, 'F'+ str(exp_id) +'_net1.pt')
torch.save(net2, 'F'+ str(exp_id) +'_net2.pt')
