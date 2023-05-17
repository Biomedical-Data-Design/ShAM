import torch
from torch import nn
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%%
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
    def forward(self, x):
        output = self.model(x)
        return output


#%%
train_data_length = 15000 #need to be dividable by batch_size
train_data = torch.zeros((train_data_length, 3))
rho01=0;rho02=0;rho12=0
sigma = [[20, rho01, rho02], [rho01, 30, rho12], [rho02, rho12, 10]]
mu = [2, 3, 4]
dstr = stats.multivariate_normal(mean=mu, cov=sigma)
x1=[];x2=[];x3=[]
for i in range(train_data_length):
    sample=dstr.rvs()
    x1.append(sample[0]);x2.append(sample[1]);x3.append(sample[2])
train_data[:, 0] = torch.tensor(x1)
train_data[:, 1] = torch.tensor(x2)
train_data[:, 2] = torch.tensor(x3)
train_labels = torch.zeros(train_data_length)
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]

batch_size =75
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)


#%%
discriminator = Discriminator()
generator = Generator()

lr = 0.001
num_epochs = 300
loss_function = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples_labels = torch.ones((batch_size, 1))
        latent_space_samples = torch.randn((batch_size, 3))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 3))

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")


#%%
latent_space_samples = torch.randn(10000, 3)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()

# ep=0.1
# g_s_x1=[]
# for i in range(len(generated_samples)):
#     if (abs(generated_samples[i,1]-3)<=ep)&(abs(generated_samples[i,2]-4)<=ep):
#         g_s_x1.append(generated_samples[i,0])

#%%
mean_x_hat_0 = np.mean(np.array(generated_samples[:, 0]))
var_x_hat_0 = np.var(np.array(generated_samples[:, 0]))
mean_x_hat_1 = np.mean(np.array(generated_samples[:, 1]))
var_x_hat_1 = np.var(np.array(generated_samples[:, 1]))
mean_x_hat_2 = np.mean(np.array(generated_samples[:, 2]))
var_x_hat_2 = np.var(np.array(generated_samples[:, 2]))

#%%
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
result_mean = np.array([mean_x_hat_0, mean_x_hat_1,mean_x_hat_2])
result_var = np.diag(np.array([var_x_hat_0, var_x_hat_1, var_x_hat_2]))
Kl_mvn = kl_mvn(np.array(mu), np.array(sigma), result_mean, result_var)

#%%

plt.figure(figsize=(15,10))

plt.suptitle('GAN analysis with KL loss = ' + str(Kl_mvn))

l1=train_data[:, 0].tolist()
plt.subplot(321)
plt.hist(l1,100,density=True)
plt.xlabel('X_1; mean=2; var=20')
plt.ylabel('Density')

l2=generated_samples[:, 0].tolist()
plt.subplot(322)
plt.hist(l2,100,density=True)
plt.xlabel('X_1_hat; mean='+str(mean_x_hat_0)+'; var='+str(var_x_hat_0)+'')

l3=train_data[:, 1].tolist()
plt.subplot(323)
plt.hist(l3,100,density=True)
plt.xlabel('X_2; mean=3; var=30')
plt.ylabel('Density')

l4=generated_samples[:, 1].tolist()
plt.subplot(324)
plt.hist(l4,100,density=True)
plt.xlabel('X_2_hat; mean='+str(mean_x_hat_1)+'; var='+str(var_x_hat_1)+'')


l5=train_data[:, 2].tolist()
plt.subplot(325)
plt.hist(l5,100,density=True)
plt.xlabel('X_3; mean=4; var=10')
plt.ylabel('Density')


l6=generated_samples[:, 2].tolist()
plt.subplot(326)
plt.hist(l6,100,density=True)
plt.xlabel('X_3_hat; mean='+str(mean_x_hat_2)+'; var='+str(var_x_hat_2)+'')

plt.show()
