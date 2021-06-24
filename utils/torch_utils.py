import torch

def sample_data(data, number, split=False):
    """Shuffle the data and randomly select number examples."""
    mx = torch.randperm(len(data), device=torch.device('cpu'))
    output = data[mx]
    if split:
        return output[number:], output[:number]
    else:
        return output[:number]


def torch2numpy(tensor):
    return tensor.detach().cpu().numpy()