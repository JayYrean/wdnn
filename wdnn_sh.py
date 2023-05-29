import os, random
import joblib
import pandas as pd
import numpy as np
from sklearn import preprocessing


from sklearn.neighbors import NearestNeighbors    # k近邻算法

class Smote:
    def __init__(self,samples,N,k):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0

    def over_sampling(self):
        N=int(self.N)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)    # 1.对每个少数类样本均求其在所有少数类样本中的k近邻
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            self._populate(N,i,nnarray)
        return self.synthetic
    # 2.为每个少数类样本选择k个最近邻中的N个；3.并生成N个合成样本
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1


def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - mu)/std


def acc(ytrue, ypred):
    correct, total, tp, np, totalP = 0, 0, 0, 0, 0
    for i, val in enumerate(ytrue):
        output = ypred[i]
        target = ytrue[i]
        if target==1:
            totalP+=1
        if output==target:
            correct += 1
        if output==target and target==1:
            tp+=1
        elif output==1:
            np+=1

        total += 1
    if totalP!=0:
        precision = tp / totalP
    else:
        precision = 0
    if tp + np != 0:
        recall = tp / (tp + np)
    else:
        recall = 0
    accuracy = correct / total
    print('Accuracy on Test Set: {:.4f} %'.format(100 * accuracy))
    print('Precision on Test Set: {:.4f} %'.format(100 * precision))
    print('recall on Test Set: {:.4f} %'.format(100 * recall))


syspath = './'

ipath = os.path.join(syspath,'shdataNoIMP.xlsx')
ip = pd.read_excel(ipath,header=0)

ipath2 = os.path.join(syspath, 'inputdata2.xlsx')
ipt = pd.read_excel(ipath2, header = 0)


X=[]
y=[]
Xtrain=[]
ytrain=[]
Xtest=[]
ytest=[]
marky=[]
splitday = 8
checklist = ['直接胆红素','总胆红素','C反应蛋白(胶体金法)','新冠抗体IgG(定量)','活化部分凝血活酶时间','肌红蛋白',
             '淋巴细胞#','血小板','白细胞','肌酐','中性粒细胞#']
mark = [[] for i in range(len(checklist)+1)]
for idx,pat in ip.iterrows():
    ls = []
    for i,check in enumerate(checklist):
        ls.append(float(pat[check]))
        mark[i].append(float(pat[check]))
    if pat['年龄'] < 60:
        ls.append(1)
        mark[-1].append(1)
    if pat['年龄'] >= 60 and pat['年龄'] < 80:
        ls.append(2)
        mark[-1].append(2)
    if pat['年龄'] >= 80:
        ls.append(3)
        mark[-1].append(3)
    if pat['实际天数']>=splitday:
       marky.append(1)
    else:
        marky.append(0)
    # y.append(pat['相差天数'])
    # X.append(np.array(ls))
mark=np.array(mark)
marky=np.array(marky)
for i in range(len(mark[0])):
    ns=[]
    for j in range(len(checklist)+1):
        ns.append(mark[j][i])
    ls = preprocessing.scale(ns)
    X.append(np.array(ns))
    y.append(marky[i])

marky2 = []
mark2 = [[] for i in range(len(checklist)+1)]
for idx,pat in ipt.iterrows():
    ls = []
    for i,check in enumerate(checklist):
        ls.append(float(pat[check]))
        mark2[i].append(float(pat[check]))
    if pat['年龄'] < 60:
        ls.append(1)
        mark2[-1].append(1)
    if pat['年龄'] >= 60 and pat['年龄'] < 80:
        ls.append(2)
        mark2[-1].append(2)
    if pat['年龄'] >= 80:
        ls.append(3)
        mark2[-1].append(3)
    if pat['实际天数']>=splitday:
       marky2.append(1)
    else:
        marky2.append(0)

mark2=np.array(mark2)
marky2=np.array(marky2)
for i in range(len(mark2[0])):
    ns=[]
    for j in range(len(checklist)+1):
        ns.append(mark2[j][i])
    ls = preprocessing.scale(ns)
    Xtest.append(np.array(ns))
    ytest.append(marky2[i])

print(np.array(X).shape, np.array(Xtest).shape)
print(np.array(X)[np.array(y) == 1].shape, np.array(Xtest)[np.array(ytest) == 1].shape)

Xtest2 = X.copy()
ytest2 = y.copy()


# X1 = np.array(X)[np.array(y) == 1]
# newX1 = Smote(X1, 2, 3).over_sampling()
# # print(np.array(X).shape, X1.shape, newX1.shape)
# X = np.append(X, newX1, axis = 0)
# # print(X.shape)
# indices = np.arange(X.shape[0])
# np.random.shuffle(indices)
# X = X[indices]
# y = np.append(y, [1 for _ in range(newX1.shape[0])])

X = np.array(X)
y = np.array(y)


Xtest1 = np.array(Xtest)[np.array(ytest) == 1]
newXtest1 = Smote(Xtest1, 2, 3).over_sampling()
Xtest = np.append(np.array(Xtest), newXtest1, axis = 0)
print(Xtest.shape)
indices = np.arange(Xtest.shape[0])
np.random.shuffle(indices)
Xtest = Xtest[indices]
ytest = np.append(ytest, [1 for _ in range(newXtest1.shape[0])])
# print(X.shape, marky.shape)
ytest = ytest[indices]


Xtest = np.array(Xtest)
ytest = np.array(ytest)


print(X.shape, Xtest.shape)
print(X[y == 1].shape, Xtest[ytest == 1].shape)

X, Xtest = X.tolist(), Xtest.tolist()
y, ytest = y.tolist(), ytest.tolist()

Xtrain = np.array(X+Xtest)
ytrain = np.array(y+ytest)

print(Xtrain.shape, ytrain.shape)


spoint = 0.8
Xlen, X2len = len(X), len(Xtest)
Xlen, X2len = int(Xlen * spoint), int(X2len * spoint)

# indices = np.arange(Xtrain.shape[0])
# np.random.shuffle(indices)
# shuffled_dataset = Xtrain[indices]
# shuffled_labels = ytrain[indices]

# Xtrain = np.array(X[:Xlen] + Xtest[:X2len])
# ytrain = np.array(y[:Xlen] + ytest[:X2len])
# Xtest = np.array(X[Xlen:] + Xtest[X2len:])
# ytest = np.array(y[Xlen:] + ytest[X2len:])


Xtrain = np.array(X[:Xlen])
ytrain = np.array(y[:Xlen])
Xtest = np.array(X[Xlen:])
ytest = np.array(y[Xlen:])

# Xtrain = np.array(Xtest[:X2len])
# ytrain = np.array(ytest[:X2len])
# Xtest = np.array(Xtest[X2len:])
# ytest = np.array(ytest[X2len:])

Xtlen = int(len(Xtest2) * spoint)

Xtest2 = np.array(Xtest2[Xtlen:])
ytest2 = np.array(ytest2[Xtlen:])

print(Xtest2.shape, ytest2.shape)

# 通过均差与标准差归一化
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
sc = sc.partial_fit(Xtest)
Xtest = sc.transform(Xtest)
Xtest2 = sc.transform(Xtest2)


# for col in range(Xtrain.shape[1]):
# #     Xtrain[:, col] = feature_normalize(Xtrain[:, col])
#     Xtrain[:, col] = np.arctan(Xtrain[:, col])

# for col in range(Xtest.shape[1]):
#     Xtest[:, col] = np.arctan(Xtest[:, col])

# for col in range(Xtest2.shape[1]):
#     Xtest2[:, col] = np.arctan(Xtest2[:, col])

# print(Xtrain, Xtest)

# trainl = int(len(shuffled_labels)*0.8)
# Xtrain = shuffled_dataset[:trainl]
# ytrain = shuffled_labels[:trainl]
# Xtest = shuffled_dataset[trainl:]
# ytest = shuffled_labels[trainl:]




import torch
import torch.nn as nn
import torch.nn.functional as F

Xtrain = torch.Tensor(Xtrain)
ytrain = torch.Tensor(ytrain)
Xtest = torch.Tensor(Xtest)
ytest = torch.Tensor(ytest)


class Balanced_CE_loss(torch.nn.Module):
    def __init__(self):
        super(Balanced_CE_loss, self).__init__()

    def forward(self, input, target):

            loss = 0.0
        # version2

            beta = 0.8
            x = torch.max(torch.log(input), torch.tensor([-100.0]))
            y = torch.max(torch.log(1-input), torch.tensor([-100.0]))
            l = -(beta*target * x + (1-beta)*(1 - target) * y)
            loss += torch.sum(l)
            return loss


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"    
    def __init__(self, alpha=.25, gamma=2):
            super(WeightedFocalLoss, self).__init__()        
            # self.alpha = torch.tensor([alpha, 1-alpha]).cuda()  
            self.alpha = torch.tensor([alpha, 1-alpha])
            self.gamma = gamma
            
    def forward(self, inputs, targets):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')        
            targets = targets.type(torch.long)
            at = self.alpha.gather(0, targets.data.view(-1)) 
            pt = torch.exp(-BCE_loss)        
            F_loss = at*(1-pt)**self.gamma * BCE_loss        
            return F_loss.mean()





input_features = 12
class XYModel(nn.Module):
    def __init__(self):
        super(XYModel,self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=c_in,out_channels=c_out,kernel_size=1,stride=1)
        # self.batchN = nn.BatchNorm1d(c_out)
        # self.Maxpool = nn.MaxPool1d(kernel_size=1,stride=1)
        self.fc = nn.Linear(in_features=input_features,out_features=18)
        # self.fc2 = nn.Linear(in_features=20,out_features=13)
        # self.fc3 = nn.Linear(in_features=20,out_features=1)
        self.fc3 = nn.Linear(in_features=18,out_features=input_features)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.norm = nn.BatchNorm1d(1)

        self.inputWeight = []

        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1),
        #     nn.BatchNorm1d(1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=1, stride=1),
        # )
    def forward(self,x):
        x1 = self.norm(x)
        x1 = self.fc(x1)
        x1= self.sig(x1)
        # x1 = self.fc2(x1)
        # x1 = self.tanh(x1)
        x2 = self.fc3(x1)
        x2 = self.sig(x2)
        # x3 = self.fc3(x2)
        # output = torch.sigmoid(x2)
        x3 = x2*x
        self.inputWeight = x3.data.detach().numpy()
        out = torch.sum(x3)
        output = torch.sigmoid(out)
        return output

def train(model,trainx,trainy,loss_func,optimizer):
    total_loss = 0

    for i,patient in enumerate(trainx):
        patient = patient.resize(1,1,input_features)
        output = model(patient)
        output = output.squeeze()
        target = trainy[i]
        loss = loss_func(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 100 == 0:
            print("Step [{}/{}] Train Loss: {:.4f}"
                  .format(i + 1, len(trainx), loss.item()))
    return total_loss / len(trainx)

def evaluate(model,testx,testy):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        tp=0
        np=0
        totalP=0
    for i,patient in enumerate(testx):
        patient = patient.resize(1,1,input_features)
        output = model(patient).data.detach().numpy().reshape(-1)[0]
        # print(output, " ")
        target = testy[i]
        if output>=0.5:
            output=1
        else:
            output=0
        if target==1:
            totalP+=1
        if output==target:
            correct += 1
        if output==target and target==1:
            tp+=1
        elif output==1:
            np+=1

        total += 1
    precision = tp/totalP
    if tp+np!=0:
     recall = tp/(tp+np)
    else:
        recall=0
    accuracy = correct / total
    print('Accuracy on Test Set: {:.4f} %'.format(100 * accuracy))
    print('Precision on Test Set: {:.4f} %'.format(100 * precision))
    print('recall on Test Set: {:.4f} %'.format(100 * recall))
    print(tp,"tp")
    return accuracy

import matplotlib.pyplot as plt
def show_curve(ys, title):
    """
    plot curlve for Loss and Accuacy
    Args:
        ys: loss or acc list
        title: loss or accuracy
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.show()

def fit(model, num_epochs, optimizer):
    """
    train and evaluate an classifier num_epochs times.
    We use optimizer and cross entropy loss to train the model.
    Args:
        model: CNN network
        num_epochs: the number of training epochs
        optimizer: optimize the loss function
    """

    # loss and optimizer
    loss_func = Balanced_CE_loss()
    # loss_func = F.binary_cross_entropy
    # loss_func = FocalLoss(2, torch.tensor([[0.32], [0.68]]))
    # loss_func = WeightedFocalLoss(0.3)


    # log train loss and test accuracy
    losses = []
    accs = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch + 1, num_epochs))
        # train step
        loss = train(model, Xtrain,ytrain, loss_func, optimizer)
        losses.append(loss)

        # evaluate step
        accuracy = evaluate(model, Xtest,ytest)
        accs.append(accuracy)
    show_curve(losses, "train loss")
    show_curve(accs, "test accuracy")


xym = XYModel()
print(xym)
lr=0.001

optimizer = torch.optim.Adam(xym.parameters(),lr=lr)
fit(xym,100,optimizer)

# print(Xtest2, ytest2)
Xtest2 = torch.Tensor(Xtest2)
ytest2 = torch.Tensor(ytest2)
print("shanghai:\n")
accuracy = evaluate(xym, Xtest2,ytest2)
print(xym.inputWeight)

# joblib.dump(xym,'cnnS+X2.pkl')
torch.save(xym, "./smote.pt")