import torch


def Optimizer(parameters, lr, weight_decay, **kwargs):
    # print('Initialised Adam optimizer')
    return torch.optim.Adam(parameters,
                            lr=float(lr),
                            weight_decay=float(weight_decay))
