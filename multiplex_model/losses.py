import torch


def nll_loss(x, mi, logvar):
    return torch.mean((x - mi) ** 2 / (torch.exp(logvar) + 1e-8) + logvar)


def beta_nll_loss(x, mi, logvar, beta=1.0):
    sg_var_beta = logvar.detach().exp().pow(beta)
    nll = (x - mi) ** 2 / (torch.exp(logvar) + 1e-8) + logvar
    beta_nll = sg_var_beta * nll
    return torch.mean(beta_nll)


def RankMe(features):
    U, S, V = torch.linalg.svd(features)
    p = S / (S.sum() + 1e-7)
    entropy = -torch.sum(p * torch.log(p + 1e-7))
    rank_me = torch.exp(entropy)
    return rank_me
