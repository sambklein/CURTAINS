import torch

def sample_data(data, number):
    """Shuffle the data and randomly select number examples."""
    mx = torch.randperm(len(data), device=torch.device('cpu'))
    output = data[mx]
    return output[:number]