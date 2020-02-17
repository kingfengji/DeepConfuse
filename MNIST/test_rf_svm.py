from sklearn.ensemble import RandomForestClassifier
import torch
from sklearn import svm

Xtrain_clean, ytrain = torch.load('data/MNIST/processed/training.pt')
Xtrain_adv = torch.load('training_adv.pt')
Xtrain_clean = Xtrain_clean.float().numpy() / 255.
ytrain = ytrain.numpy()
Xtrain_adv = Xtrain_adv.numpy()

Xtest_clean, ytest = torch.load('data/MNIST/processed/test.pt')
Xtest_adv = torch.load('test_adv.pt')
Xtest_clean = Xtest_clean.float().numpy() / 255.
ytest = ytest.numpy()
Xtest_adv = Xtest_adv.numpy()

print('=============SVM===============')
svc = svm.SVC(C=200, kernel='rbf', gamma=0.01, cache_size=8000,
              probability=False)
svc.fit(Xtrain_adv.reshape(60000, -1), ytrain)
print('train on adv, test on clean: ',
      svc.score(Xtest_clean.reshape(10000, -1), ytest))
svc = svm.SVC(C=200, kernel='rbf', gamma=0.01, cache_size=8000,
              probability=False)
svc.fit(Xtrain_clean.reshape(60000, -1), ytrain)
print('train on clean, test on clean: ',
      svc.score(Xtest_clean.reshape(10000, -1), ytest))

print('=============RF===============')
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100)
rfc.fit(Xtrain_adv.reshape(60000, -1), ytrain)
print('train on adv, test on clean: ',
      rfc.score(Xtest_clean.reshape(10000, -1), ytest))
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100)
rfc.fit(Xtrain_clean.reshape(60000, -1), ytrain)
print('train on clean, test on clean: ',
      rfc.score(Xtest_clean.reshape(10000, -1), ytest))

import IPython;

IPython.embed()
