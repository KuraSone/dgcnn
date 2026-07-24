"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def cal_loss(pred: Tensor, gold: Tensor, smoothing: bool = True) -> Tensor:
    """ Calculate cross entropy loss, apply label smoothing if needed. """

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream:
    def __init__(self, path: str) -> None:
        self.f = open(path, 'a')

    def cprint(self, text: str) -> None:
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self) -> None:
        self.f.close()
