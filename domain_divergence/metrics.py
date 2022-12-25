from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils import create_discrete_distribution, create_continuous_gaussian_distribution
import torch

def kl_divergence_discrete(target_features, source_features):


    source = create_discrete_distribution(source_features)
    target = create_discrete_distribution(target_features)
    
    all_values = {}
    cnt = 0
    for value in source.keys():
        if value not in all_values.keys():
            all_values[cnt] = value
            cnt += 1

    for value in target.keys():
        if value not in all_values.keys():
            all_values[cnt] = value
            cnt += 1

    source_distribution = torch.ones(cnt)
    target_distribution = torch.ones(cnt)

    for i in range(cnt):

        value = all_values[i]

        if value in source.keys():
            source_distribution[i] += source[value]
        
        if value in target.keys():
            target_distribution[i] += target[value]

    
    return kl_div(target_distribution, source_distribution).mean()

def js_divergence_discrete(target, source):
    return (kl_divergence_discrete(target, source) + kl_divergence_discrete(source, target))/2


def pad(target_features, source_features, model, epochs, device):


    features = torch.cat((target_features, source_features), dim = 0)


    labels_1 = torch.zeros(target_features.shape[0])
    labels_2 = torch.ones(source_features.shape[0])

    labels = torch.cat((labels_1, labels_2), dim = 0)


    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    
    model = model.to(device)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()

        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            batch = tuple(t.to(device).to(torch.float32) for t in batch)
            input_ids, label_ids = batch
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            label_ids = label_ids.to(torch.long)
            with torch.set_grad_enabled(True):
                output = model(input_ids)
                loss = loss_fn(output, label_ids)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

    model.eval()
    loss = 0
    steps = 0
    for batch in tqdm(dataloader, desc="Iteration"):
        batch = tuple(t.to(device).to(torch.float32) for t in batch)
        input_ids, label_ids = batch
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)
        label_ids = label_ids.to(torch.long).to(device)
        with torch.set_grad_enabled(False):
            logits = model(input_ids)
            loss_cur = loss_fn(logits, label_ids)
            
            loss += loss_cur.cpu().item()
            steps += input_ids.shape[0]
    
    return loss/steps


def snr(feature_target, feature_source):

    last = feature_target.shape[-1]
    feature_target_updated = feature_target.reshape(-1, last).squeeze()
    feature_source_updated = feature_source.reshape(-1, last).squeeze()
    target_var = torch.var(feature_target_updated, dim = 0)
    diff = feature_target_updated - feature_source_updated
    diff_var = torch.var(diff, dim=0)

    snr = target_var/diff_var

    return snr.norm()

def kl_divergence_continuous(source_features, target_features):

    last = target_features.shape[-1]

    feature_target_updated = target_features.reshape(-1, last).squeeze()
    feature_source_updated = source_features.reshape(-1, last).squeeze()

    mean_source, cov_source = create_continuous_gaussian_distribution(feature_source_updated)
    mean_target, cov_target = create_continuous_gaussian_distribution(feature_target_updated)

    
    if mean_source.shape == torch.tensor(1).shape:
        return 0.5*(torch.log(cov_target/cov_source) + (cov_source*cov_source + (mean_source - mean_target)*(mean_source - mean_target))/(cov_target*cov_target))

    d = cov_source.shape[0]

    final_term = torch.matmul(torch.matmul((mean_source - mean_target).T, torch.inverse(cov_target)), (mean_source - mean_target))

    print(torch.slogdet(torch.matmul(cov_target, torch.inverse(cov_source))).logabsdet)
    
    return 0.5*(torch.slogdet(torch.matmul(cov_target, torch.inverse(cov_source))).logabsdet + torch.trace(torch.mm(torch.inverse(cov_target), cov_source)) - d + final_term)
    

def js_div_continuous(mean_source, cov_source, mean_target, cov_target):

    return (kl_divergence_continuous(mean_source, cov_source, mean_target, cov_target) + kl_divergence_continuous(mean_target, cov_target, mean_source, cov_source))/2