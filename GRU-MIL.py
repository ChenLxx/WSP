#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torchviz import make_dot, make_dot_from_trace
import torch.utils.data as Data

import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np
import util

import glob


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0，1"
util.setup_log()
use_cuda = torch.cuda.is_available()


# In[2]:


#DATA
X_raw=np.zeros((86400,11))
dat1 = pd.read_csv('/home/liuchen/wind_data/K111+340_2017-09-30.txt',header=None)
dat2 = pd.read_csv('/home/liuchen/wind_data/K117+378_2017-09-30.txt',header=None)
dat3 = pd.read_csv('/home/liuchen/wind_data/K127+995_2017-09-30.txt',header=None)
dat4 = pd.read_csv('/home/liuchen/wind_data/K135+913_2017-09-30.txt',header=None)
dat5 = pd.read_csv('/home/liuchen/wind_data/K145+688_2017-09-30.txt',header=None)
dat6 = pd.read_csv('/home/liuchen/wind_data/K154+192_2017-09-30.txt',header=None)
dat7 = pd.read_csv('/home/liuchen/wind_data/K164+845_2017-09-30.txt',header=None)
dat8 = pd.read_csv('/home/liuchen/wind_data/K174+468_2017-09-30.txt',header=None)
dat9 = pd.read_csv('/home/liuchen/wind_data/K186+682_2017-09-30.txt',header=None)
dat10= pd.read_csv('/home/liuchen/wind_data/K196+146_2017-09-30.txt',header=None)
dat11= pd.read_csv('/home/liuchen/wind_data/K203+320_2017-09-30.txt',header=None)

X_raw[:,0]=dat5.iloc[:,1].values
X_raw[:,1]=dat7.iloc[:,1].values
X_raw[:,2]=dat4.iloc[:,1].values
X_raw[:,3]=dat8.iloc[:,1].values
X_raw[:,4]=dat3.iloc[:,1].values
X_raw[:,5]=dat9.iloc[:,1].values
X_raw[:,6]=dat2.iloc[:,1].values
X_raw[:,7]=dat10.iloc[:,1].values
X_raw[:,8]=dat1.iloc[:,1].values
X_raw[:,9]=dat11.iloc[:,1].values
X_raw[:,10]=dat6.iloc[:,1].values

def Select_data(X,ctime):
    row_old=len(X)
    colunm=len(X[0])
    row_new=int(row_old/ctime)
    X_select=np.zeros((row_new,colunm))
    for i in range(colunm):
        for j in range(row_new):
            X_select[j,i]=np.max(X[j*ctime:j*ctime+ctime,i])
    return X_select

X_select=Select_data(X_raw,60)
print(X_select.shape)
#driven series
X0=X_select[:,0:10]
Y0=X_select[:,10]


# In[3]:


BATCH_SIZE=100
TRAINSIZE=720
EPOCH=4000
T=20


# In[4]:


X1= X_select[0:1020,0:8]
#targt series
Y1= X_select[0:1020,10]
ROW=X1.shape[0]
COLUMN=X1.shape[1]
'''
X1=np.zeros((len(X0),len(X0[0])))
for i in range(len(X0[0])):
    X1[:,i]=(X0[:,i]-np.amin(X0[:,i]))/(np.amax(X0[:,i])-np.amin(X0[:,i]))
Y1=(Y0-np.amin(Y0))/(np.amax(Y0)-np.amin(Y0))
'''
print(Y1.shape)
#targt label
y_mean=np.mean(X_select)
#y=np.zeros(len(Y1))
#y[Y1>y_mean]=1
#mu = np.mean(Y1,axis=0)
#std = np.std(Y1,axis=0)
#Ymax=np.amax(Y1)
#Ymin=np.amin(Y1)
#Y=(Y1-Ymin)/(Ymax-Ymin)
#print(y.shape)
plt.figure()
plt.plot(range(1, 1+len(Y1)), Y1, label = "True")
plt.legend(loc = 'upper left')
plt.show()
plt.figure()
#plt.savefig('G:\\导出的图片.png')#保存图片
print(X1)
print(Y1)


# In[5]:


y_mean=np.mean(X_select)
print(y_mean)
y_label=np.zeros((len(Y0)-T))
for i in range(len(Y0)-T):
    for j in range(T):
        if Y0[i+j]>y_mean*1.45:
            y_label[i]=1
            break
y_label_valid=y_label[TRAINSIZE-T:len(Y1)-T]
y_label_train=y_label[:TRAINSIZE-T]
y_label_test=y_label[len(Y1)-T:]
#print(y_label_vali.shape)
#print(y_pred_vali.shape)
#print(y_label_vali.shape)
plt.figure()
plt.plot(range(1, 1+len(y_label_train)), y_label_train, label = "Train")
plt.plot(range(1+len(y_label_train), 1+len(y_label_train)+len(y_label_valid)), y_label_valid, label = "valid")
plt.plot(range(1+len(y_label_train)+len(y_label_valid), 1+len(y_label)), y_label_test, label = "test")
plt.legend(loc = 'upper left')
plt.show()
print(y_label.shape)


# In[6]:


class gru_mil(nn.Module):
    def __init__(self,in_dim,T,gru_size,fc_hidden_size,out_dim):
        super(gru_mil, self).__init__()
        self.T=T
        self.gru_size=gru_size
        self.rnn=nn.GRU(in_dim,gru_size,1,batch_first=True)
        self.fc = nn.Sequential(nn.Linear(gru_size, fc_hidden_size),nn.Dropout(0.5) ,nn.Tanh()).cuda()

        self.logitic = nn.Sequential(nn.Linear(fc_hidden_size, out_dim), nn.Sigmoid()).cuda()
        
    def forward(self,X):
        r_out,h_state=self.rnn(X,None)
        #mil
        out_fc = self.fc(r_out)
        out_p = self.logitic(out_fc)
        p_max = torch.max(out_p, 1)[0].cuda()
        return p_max


# In[7]:


# network parameter
in_dim = len(X1[0])
out_dim = 1
gru_size = 32  # GRU size
fc_hidden_size = 50


# In[25]:


model = gru_mil(in_dim,T,gru_size,fc_hidden_size,out_dim).cuda()


# In[26]:


weight_p,bias_p=[],[]
for name,p in model.named_parameters():
    if 'bias' in name:
        bias_p.append(p)
    else:
        weight_p.append(p)


# In[27]:


k1=0.001
learning_rate=0.002
criterion = torch.nn.BCELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0110)
optimizer=torch.optim.Adam([{ 'params':weight_p,'weight_decay':0.001},
                            { 'params':bias_p,'weight_decay':0}],
                            lr=learning_rate
)


# In[28]:


def predict(on_train = False):
        if on_train:
            y_pred = np.zeros(TRAINSIZE - T )
        else:
            y_pred = np.zeros(ROW - TRAINSIZE)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + BATCH_SIZE)]
            X = np.zeros((len(batch_idx), T - 1, COLUMN))
            y_history = np.zeros((len(batch_idx),T - 1))
            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = X1[range(batch_idx[j], batch_idx[j] + T - 1), :]
                    y_history[j, :] = Y1[range(batch_idx[j],  batch_idx[j]+ T - 1)]
                else:
                    X[j, :, :] = X1[range(batch_idx[j] + TRAINSIZE - T, batch_idx[j] + TRAINSIZE - 1), :]
                    y_history[j, :] = Y1[range(batch_idx[j] + TRAINSIZE - T,  batch_idx[j]+ TRAINSIZE - 1)]

            #_, input_encoded = encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()))
            #input_final=decoder(input_encoded,Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda()))
            #print(input_final)
            #print(mil(input_final).cpu().data.numpy()[:, 0])
            y_pred[i:(i + BATCH_SIZE)] = model(Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()),).cpu().data.numpy()[:, 0]
            #print('here1')
            #y_pred[i:(i + BATCH_SIZE)] = mil(Variable(torch.from_numpy(input_final).type(torch.FloatTensor).cuda()))
            #y_pred.append(mil(input_final))
            i += BATCH_SIZE
        return y_pred


# In[29]:


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# In[30]:


train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []


# In[31]:


patience=15
early_stopping = EarlyStopping(patience=patience, verbose=True)


# In[35]:


#torch.save(model.state_dict())
n_iter=0
#train
y_target=np.zeros((BATCH_SIZE,1))
for epoch in range(EPOCH):
    #perm_idx = np.random.permutation(TRAINSIZE - T)
    perm_idx=np.array(range(TRAINSIZE-T))
    j = 0
    while j < (TRAINSIZE-T):
        batch_idx = perm_idx[j:(j + BATCH_SIZE)]
        X_history = np.zeros((len(batch_idx), T - 1 , X1.shape[1]))
        Y_history = np.zeros((len(batch_idx), T - 1))
        y_target[:,0] =y_label[batch_idx]

        for k in range(len(batch_idx)):
            X_history[k, :, :] = X1[batch_idx[k] : (batch_idx[k] + T- 1 ), :]
            Y_history[k, :] = Y1[batch_idx[k] : (batch_idx[k] + T - 1)]
            
        optimizer.zero_grad()

        y_pred = model(Variable(torch.from_numpy(X_history).type(torch.FloatTensor).cuda()))
        y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor).cuda())
        l1_reg=0
        for param in model.parameters():
            l1_reg+=torch.sum(torch.abs(param))
        loss = criterion(y_pred, y_true)
        total_loss=loss+k1*l1_reg
        total_loss.backward()
        #print(loss)
        optimizer.step()
        #train_losses.append(loss.item())
        j += BATCH_SIZE
    #'''
        n_iter+=1
        if n_iter % 200 == 0 and n_iter > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
    #'''
    #calculate the vaidation loss
    
    y_pred_valid = predict(on_train = False)
    loss_valid = criterion(Variable(torch.from_numpy(y_pred_valid).type(torch.FloatTensor).cuda()), 
                         Variable(torch.from_numpy(y_label_valid).type(torch.FloatTensor).cuda()))
    # record validation loss
    valid_losses.append(loss_valid.item())
    valid_loss_iter=loss_valid.item()
    
    #calculate the train loss
    y_pred_train = predict(on_train = True)   
    loss_train = criterion(Variable(torch.from_numpy(y_pred_train).type(torch.FloatTensor).cuda()), 
                         Variable(torch.from_numpy(y_label_train).type(torch.FloatTensor).cuda()))
    # record validation loss
    train_losses.append(loss_train.item())
    train_loss_iter=loss_train.item()
    
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    '''
    early_stopping(valid_loss_iter, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    #'''
    if epoch % 10 == 0:
        print('epoch=',epoch+10)
        print('loss=', loss)
    if epoch % 10 == 0:
        y_train_pred = predict(on_train = True)
        y_test_pred = predict(on_train = False)
        y_pred = np.concatenate((y_train_pred, y_test_pred))
        plt.figure()
        plt.plot(range(1+T, 1 + T+len(y_label_train)), y_label_train, label = "True")
        plt.plot(range(1+T+len(y_label_train), 1 + T+len(y_label_train)+len(y_label_valid)), y_label_valid, label = "True")
        plt.plot(range(T , len(y_train_pred) + T), y_train_pred, label = 'Predicted - Train')
        plt.plot(range(T + len(y_train_pred) , len(Y1) ), y_test_pred, label = 'Predicted - Test')
        plt.legend(loc = 'upper left')
        plt.show()
        


# In[33]:


plt.figure()
plt.plot(range(1, 1+len(train_losses)), train_losses, label = "loss")
plt.legend(loc = 'upper left')
plt.show()


# In[34]:


plt.figure()
plt.plot(range(1, 1+len(valid_losses)), valid_losses, label = "loss")
plt.legend(loc = 'upper left')
plt.show()


# In[272]:


print(np.amin(valid_losses[-5*T:-1]))
print(np.amin(train_losses[-5*T:-1]))


# In[268]:


X_test= X_select[1000:1420,0:8]
#targt series
Y_test= X_select[1000:1420,10]
y_test_label=y_label[1000:1400]
print(Y_test.shape)
plt.figure()
plt.plot(range(1+T, 1 + T+len(y_test_label)), y_test_label, label = "True-Test")
plt.legend(loc = 'upper left')
plt.show()


# In[269]:


y_test_pred=np.zeros((len(y_test_label)))
i=0
while i < len(y_test_pred):
        batch_idx = np.array(range(len(y_test_pred)))[i : (i + BATCH_SIZE)]
        X = np.zeros((len(batch_idx), T - 1, COLUMN))
        y_history = np.zeros((len(batch_idx),T - 1))
        for j in range(len(batch_idx)):
            X[j, :, :] = X_test[range(batch_idx[j], batch_idx[j] + T - 1), :]
            y_history[j, :] = Y_test[range(batch_idx[j],  batch_idx[j]+ T - 1)]
            
        

            #_, input_encoded = encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()))
            #input_final=decoder(input_encoded,Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda()))
            #print(input_final)
            #print(mil(input_final).cpu().data.numpy()[:, 0])
        y_test_pred[i:(i + BATCH_SIZE)] = model(Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()),).cpu().data.numpy()[:, 0]
            #print('here1')
            #y_pred[i:(i + BATCH_SIZE)] = mil(Variable(torch.from_numpy(input_final).type(torch.FloatTensor).cuda()))
            #y_pred.append(mil(input_final))
        i += BATCH_SIZE


# In[270]:


plt.figure()
plt.plot(range(1+T, 1 + T+len(y_test_label)), y_test_label, label = "True")
plt.plot(range(T , len(y_test_pred) + T), y_test_pred, label = 'Predicted - Test')
plt.legend(loc = 'upper left')
plt.show()


# In[271]:


from math import sqrt
N=len(y_test_label)
error = []
for i in range(N):
    error.append(y_test_label[i] - y_test_pred[i])
squaredError = []
absError = []
absperError=[]
for i in range(len(error)):
    squaredError.append(error[i] * error[i])#target-prediction之差平方 
    absError.append(abs(error[i]))
    absperError.append(abs(error[i])/abs(y_test_label[i]))
MAE=sum(absError)/N
MAPE=sum(absperError)/N
MSE=sum(squaredError)/N
print('MAE=',MAE)
#print('MAPE=',MAPE)
print('MSE=',MSE)


# In[5]:


import numpy as np
import matplotlib.pyplot as plt

plt.axis([0,200,0,100])
x=np.arange(0,200,5)
y=np.random.randint(20,80,len(x))

for index in range(len(x)-1):
    plt.plot([x[index],x[index+1]],[y[index],y[index+1]])
    plt.pause(0.2)
plt.show


# In[ ]:




