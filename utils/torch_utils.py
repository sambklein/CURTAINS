import torch

def sample_data(data, number, split=False):
    """Shuffle the data and randomly select number examples."""
    mx = torch.randperm(len(data), device=torch.device('cpu'))
    output = data[mx]
    if split:
        return output[number:], output[:number]
    else:
        return output[:number]


def shuffle_tensor(data):
    mx = torch.randperm(len(data), device=torch.device('cpu'))
    return data[mx]


def tensor2numpy(x):
    # try:
    #     return tensor.detach().cpu().numpy()
    # except Exception as e:
    #     print('Transform from tensor to numpy failed. Returning object directly.')
    #     return tensor
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x