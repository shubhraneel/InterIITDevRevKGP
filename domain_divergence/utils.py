import torch

def create_discrete_distribution(features):
    '''
    input: torch.tensor
    output: dict[torch.tensor] --> freq
    '''
    dist = {}

    for point in features:
        if point not in dist.keys():
            dist[point] = 0
        dist[point] += 1

    return dist


def create_continuous_gaussian_distribution(features):
    '''
    input: torch.tensor
    output: mean(torch.tensor), covariance_matrix(torch.tensor)
    '''

    features.to(torch.float32)
    mean = torch.mean(features, dim = 0)

    reshapped_features = features.unsqueeze(-1).transpose(0, -1).squeeze(0)
    covariance_matrix = torch.cov(reshapped_features)
    return mean, covariance_matrix