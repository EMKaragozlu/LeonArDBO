import pandas as pd
from gp_model import ExactGP
import torch
from sklearn.preprocessing import MinMaxScaler
import copy

dataset_name = 'AgNP'
negative = True # maximize or minimize the objective function
N = 10000000 # number of samples to draw from the prior

raw_dataset = pd.read_csv('Benchmarking\\datasets\\' + dataset_name + '_dataset.csv')
feature_name = list(raw_dataset.columns)[:-1]
objective_name = list(raw_dataset.columns)[-1]

if negative:
    ds = copy.deepcopy(raw_dataset) 
    ds[objective_name] = -raw_dataset[objective_name].values
else:
    ds = raw_dataset

ds_grouped = ds.groupby(feature_name)[objective_name].agg(lambda x: x.unique().mean())
ds_grouped = (ds_grouped.to_frame()).reset_index()
X_feature = ds_grouped[feature_name].values

scaler = MinMaxScaler()
X_feature = scaler.fit_transform(X_feature)
X_feature = torch.Tensor(X_feature)

y = torch.Tensor(ds_grouped[objective_name].values)

#get the prior mean and covariance
model = ExactGP(X_feature, y)
model.likelihood.noise_covar.noise = 1e-2
model.likelihood.noise_covar.raw_noise.requires_grad = False
model.covar_module.base_kernel.lengthscale = 1.0
model.covar_module.outputscale = 1.0

posterior = model(X_feature)

assert len(ds_grouped) == len(X_feature) == len(y)
# total number of data in set
M = len(ds_grouped)
samples = posterior.sample(sample_shape=torch.Size([N]))
argmaxes = torch.argmax(samples, axis=1)
unique, counts = torch.unique(argmaxes, return_counts=True)

def argmin_all(tensor):
    min_value = tensor.min()
    a = torch.where(tensor == min_value)[0]
    return tuple(int(t.item()) for t in a)

def argmax_all(tensor):
    max_value = tensor.max()
    a = torch.where(tensor == max_value)[0]
    return tuple(int(t.item()) for t in a)

corners = [set() for _ in range(X_feature.shape[1])]

for j in range(X_feature.shape[1]):
    argmins = set(argmin_all(X_feature[:, j]))
    argmaxs = set(argmax_all(X_feature[:, j]))
    corners[j].update(argmins)
    corners[j].update(argmaxs)

cornership_freq = torch.zeros(X_feature.shape[0])
for idx in range(X_feature.shape[0]):
    for j in range(X_feature.shape[1]):
        if idx in corners[j]:
            cornership_freq[idx] += 1

_, inverse_indices = torch.unique(cornership_freq, sorted=True, return_inverse=True)

for threshold in range(1, int(torch.max(cornership_freq)) + 1):
    corner_prob = 0
    for c in range(X_feature.shape[0]):
        if cornership_freq[c] >= threshold:# torch.max(cornership_freq):
            corner_prob += counts[c]/N

    print('Corner probability is ', float(corner_prob)*100, ' percent for threshold', threshold)

    actual_argmax = torch.argmax(y)
    if cornership_freq[actual_argmax] >= threshold:# torch.max(cornership_freq):
        print('The actual argmax is a {} corner'.format(threshold))
    else:
        print('The actual argmax is not a {} corner'.format(threshold))