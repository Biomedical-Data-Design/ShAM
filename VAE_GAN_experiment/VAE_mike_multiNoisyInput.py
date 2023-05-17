# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:51:36 2022

@author: quick2063706271 
"""

import numpy as np
from sklearn import preprocessing

#%%
mean = np.array([2, 3, 4])
cov = [[20, 0,0], [0, 30,0], [0,0,10]]  # diagonal covariance

dataset_size = 20000

dataset = np.random.multivariate_normal(mean, cov, dataset_size)
dataset = dataset.astype(np.float32)
dim = dataset.shape[1]

#%%
# make sure of good reconstruction
def mse_loss(y_pred, y_true):
    loss = nn.MSELoss(reduction='sum', size_average=False)
    return loss(y_pred, y_true)

# make sure that the latent space is continuous and standard normal distributed
def kld_Loss(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

# loss functino of the VAE
def loss_function(y_pred, y_true, input_dim):
    recon_x, mu, logvar = y_pred
    x = y_true
    KLD = kld_Loss(mu, logvar)
    MSE = mse_loss(recon_x, x)
    return KLD + MSE

def kl_divergence(mean1, var1, mean2, var2):
    return np.log(var2/var1) + (var1 ** 2 + (mean1 - mean2) ** 2) / (2 * var2 ** 2) - 1/2


def kl_mvn(m0, S0, m1, S1):
    """
    https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
    The following function computes the KL-Divergence between any two 
    multivariate normal distributions 
    (no need for the covariance matrices to be diagonal)
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    - accepts stacks of means, but only one S0 and S1
    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    # 'diagonal' is [1, 2, 3, 4]
    tf.diag(diagonal) ==> [[1, 0, 0, 0]
                          [0, 2, 0, 0]
                          [0, 0, 3, 0]
                          [0, 0, 0, 4]]
    # See wikipedia on KL divergence special case.              
    #KL = 0.5 * tf.reduce_sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=1)   
                if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))                               
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 



#%%
###########################################################################################################
#                                               model defination                                          #
###########################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms

class VAE(nn.Module):
    def __init__(self, zdim, input_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, 20)
        self.fc21 = nn.Linear(20, zdim) 
        self.fc22 = nn.Linear(20, zdim) 
        self.fc3 = nn.Linear(zdim, 20)
        self.fc4 = nn.Linear(20, input_dim)
        self.input_dim = input_dim
    # encoder
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    # generating the latent layer values
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn(mu.size(0),mu.size(1)) # assume eps normally distributed ~N(0,1)
            z = mu+ eps*std
            return z
        
    # decoder
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    

#%%
def train(model, dataset, num_epochs = 1, batch_size = 64, learning_rate = 0.0002):
    model.train() #train mode
    torch.manual_seed(42)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    optimizer = optim.Adam(model.parameters(), learning_rate)
    
    for epoch in range(num_epochs):
      for data in train_loader:  # load batch
          recon_mu_logvar = model(data) # recon_mu_logvar contains recon_x, mu, and logvar
          loss = loss_function(recon_mu_logvar, data, 5) # calculate loss
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
      # print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
  
def generate(num_data_generated,latent_shape_0, latent_shape_1, VAE_model):
    result = []
    for i in range(num_data_generated):
      rinpt = torch.randn(latent_shape_0, latent_shape_1)
      with torch.no_grad():
        si = VAE_model.decode(rinpt).numpy()
      result.append(si)
    return result
#%%
'''
training starts
'''
batch_size = 256
latent_size = 3
input_size = 3
num_epochs = 300 
lr = 0.005

model = VAE(latent_size, input_size)
train(model, dataset, num_epochs = num_epochs, batch_size = batch_size, learning_rate = lr)
#%%
'''
generate with model   
'''
num_data_generated = 10000
latent_shape_0 = 1
latent_shape_1 = 3
VAE_model = model
result = generate(num_data_generated,latent_shape_0, latent_shape_1, VAE_model)

#%%
result_new = np.array(result)[:,0,:]

#%%
'''
var and mean calculation for dataset 
'''
var_x_hat_0 = result_new[:,0].var()
mean_x_hat_0 = result_new[:,0].mean()
var_x_hat_1 = result_new[:,1].var()
mean_x_hat_1 = result_new[:,1].mean()
var_x_hat_2 = result_new[:,2].var()
mean_x_hat_2 = result_new[:,2].mean()

result_mean = np.array([mean_x_hat_0, mean_x_hat_1,mean_x_hat_2])
result_var = np.diag(np.array([var_x_hat_0, var_x_hat_1, var_x_hat_2]))

#%%
def noisy_data_VAE(noise_mean,noise_cov, dataset_size,dataset, latent_size, input_size, num_epochs,batch_size, lr, num_data_generated, latent_shape_0, latent_shape_1, result_new):
    noise = np.random.multivariate_normal(noise_mean, noise_cov, dataset_size)
    noisy_dataset = dataset + noise 
    noisy_dataset = noisy_dataset.astype(np.float32)
    '''
    training with noisy data starts   
    '''
    
    noisy_model = VAE(latent_size, input_size)
    train(noisy_model, noisy_dataset, num_epochs = num_epochs , batch_size = batch_size, learning_rate = lr)
    
    '''
    generate with noisy model 
    '''
    VAE_noisy_model = noisy_model
    noisy_result = generate(num_data_generated,latent_shape_0, latent_shape_1, VAE_noisy_model)
    result_new_noisy = np.array(noisy_result)[:,0,:]
    # print(result_new_noisy.shape)
    
    '''
    var and mean calculation for noisy dataset 
    '''
    noisy_var_x_hat_0 = result_new_noisy[:,0].var()
    noisy_mean_x_hat_0 = result_new_noisy[:,0].mean()
    noisy_var_x_hat_1 = result_new_noisy[:,1].var()
    noisy_mean_x_hat_1 = result_new_noisy[:,1].mean()
    noisy_var_x_hat_2 = result_new_noisy[:,2].var()
    noisy_mean_x_hat_2 = result_new_noisy[:,2].mean()
    
    noisy_result_mean = np.array([noisy_mean_x_hat_0, noisy_mean_x_hat_1,noisy_mean_x_hat_2])
    # print(noisy_result_mean)
    noisy_result_var = np.diag(np.array([noisy_var_x_hat_0, noisy_var_x_hat_1, noisy_var_x_hat_2]))
    # print(noisy_result_var)
    return noisy_result_mean, noisy_result_var, noisy_model, result_new_noisy, noisy_dataset


#%%
###########################################################################################################
#                                         Plot the simulation data                                        #
###########################################################################################################
def plot_data(dataset,noisy_dataset,noise_mean,noise_cov,result_new,result_new_noisy):
    
    noisy_var_x_hat_0 = result_new_noisy[:,0].var()
    noisy_mean_x_hat_0 = result_new_noisy[:,0].mean()
    noisy_var_x_hat_1 = result_new_noisy[:,1].var()
    noisy_mean_x_hat_1 = result_new_noisy[:,1].mean()
    noisy_var_x_hat_2 = result_new_noisy[:,2].var()
    noisy_mean_x_hat_2 = result_new_noisy[:,2].mean()
    

    '''
    X1
    '''
    plt.figure(figsize=(20,4))
    plt.suptitle('VAE using X_1 with and without noise')
    l1=dataset[:, 0].tolist()
    plt.subplot(141)
    plt.hist(l1,100,density=True)
    plt.xlabel('X_1; mean=2; var=20')
    plt.ylabel('Density')
    
    l2=noisy_dataset[:, 0].tolist()
    plt.subplot(142)
    plt.hist(l2,100,density=True)
    plt.xlabel('X_1 with noise(mean = '+str(noise_mean[0])+', var = '+str(noise_cov[0])+')')
    plt.ylabel('Density')
    
    
    l3=result_new[:, 0].tolist()
    plt.subplot(143)
    plt.hist(l3,100,density=True)
    plt.xlabel('X_1_hat; mean='+str(mean_x_hat_0)+'; var='+str(var_x_hat_0)+'')
    
    l4 = result_new_noisy[:,0].tolist()
    plt.subplot(144)
    plt.hist(l4, 100, density=True)
    plt.xlabel('X_1_hat_noisy; mean='+str(noisy_mean_x_hat_0)+'; var='+str(noisy_var_x_hat_0)+'')
    
    
    '''
    X2
    '''
    plt.figure(figsize=(20,4))
    plt.suptitle('VAE using X_2 with and without noise')
    l5=dataset[:, 1].tolist()
    plt.subplot(141)
    plt.hist(l5,100,density=True)
    plt.xlabel('X_2; mean=3; var=30')
    plt.ylabel('Density')
    
    l6=noisy_dataset[:, 1].tolist()
    plt.subplot(142)
    plt.hist(l6,100,density=True)
    plt.xlabel('X_2 with noise(mean = '+str(noise_mean[1])+', var = '+str(noise_cov[1])+')')
    plt.ylabel('Density')
    
    l7=result_new[:, 1].tolist()
    plt.subplot(143)
    plt.hist(l7,100,density=True)
    plt.xlabel('X_2_hat; mean='+str(mean_x_hat_1)+'; var='+str(var_x_hat_1)+'')
    
    
    l8 = result_new_noisy[:,1].tolist()
    plt.subplot(144)
    plt.hist(l8, 100, density=True)
    plt.xlabel('X_2_hat_noisy; mean='+str(noisy_mean_x_hat_1)+'; var='+str(noisy_var_x_hat_1)+'')
    
    
    '''
    X3
    '''
    
    plt.figure(figsize=(20,4))
    plt.suptitle('VAE using X_3 with and without noise')
    l9=dataset[:, 2].tolist()
    plt.subplot(141)
    plt.hist(l9,100,density=True)
    plt.xlabel('X_3; mean=4; var=10')
    plt.ylabel('Density')
    
    l10=noisy_dataset[:, 2].tolist()
    plt.subplot(142)
    plt.hist(l10,100,density=True)
    plt.xlabel('X_3 with noise(mean = '+str(noise_mean[2])+', var = '+str(noise_cov[2])+')')
    plt.ylabel('Density')
    
    l11=result_new[:, 2].tolist()
    plt.subplot(143)
    plt.hist(l11,100,density=True)
    plt.xlabel('X_3_hat; mean='+str(mean_x_hat_2)+'; var='+str(var_x_hat_2)+'')
    
    
    l12 = result_new_noisy[:,2].tolist()
    plt.subplot(144)
    plt.hist(l12, 100, density=True)
    plt.xlabel('X_3_hat_noisy; mean='+str(noisy_mean_x_hat_2)+'; var='+str(noisy_var_x_hat_2)+'')
    
    plt.show()


def construct_cov_3(cov):
    return [[cov, 0,0], [0, cov,0], [0,0,cov]]
#%%
###########################################################################################################
#                                         simulation analysis                                             #
###########################################################################################################
noise_mean = np.array([0,0,0])
# noise_cov = [[10, 0,0], [0, 10,0], [0,0,10]]
noise_cov_list = [0.0001, 0.001, 0.01, 0.1, 1, 10,100,1000]
kl_mvn_list = []
for a_cov in noise_cov_list:
    print(a_cov)
    noise_cov = construct_cov_3(a_cov)
    noisy_result_mean, noisy_result_var, noisy_model, result_new_noisy, noisy_dataset= noisy_data_VAE(noise_mean,noise_cov, dataset_size,dataset, latent_size, input_size, num_epochs,batch_size, lr, num_data_generated, latent_shape_0, latent_shape_1, result_new)
    plot_data(dataset,noisy_dataset,noise_mean,noise_cov,result_new,result_new_noisy)
    Kl_mvn_noisy = kl_mvn(mean, cov, noisy_result_mean, noisy_result_var)
    kl_mvn_list.append(Kl_mvn_noisy)

#%%
Kl_mvn = kl_mvn(mean, cov, result_mean, result_var)
# Kl_mvn_noisy = kl_mvn(mean, cov, noisy_result_mean, noisy_result_var)

plt.figure(figsize=(7,7))
plt.plot(np.log10(noise_cov_list),kl_mvn_list)
plt.xlabel('$\sigma^2$ for $\epsilon$ ($\log_{10}$)')
plt.ylabel('KL test statistic')
plt.title('Effect of error variance in training data on KS statistic')