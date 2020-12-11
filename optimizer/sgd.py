import torch


def Optimizer(parameters, lr, weight_decay, **kwargs):
    # print('Initialised SGD optimizer')
    return torch.optim.SGD(parameters,
                           lr=float(lr),
                           momentum=0.9,
                           weight_decay=float(weight_decay))
