from typing import List
import torch

def binary_accuracy(output: torch.Tensor, target: torch.Tensor, compute_sigmoid: bool = False) -> float:
    """Adapted from `~tllib.utils.metric.binary_accuracy`.
    
    Computes the accuracy for binary classification
    If `compute_sigmoid` is True, the output is assumed to be logits and will be passed through 
    a sigmoid function.
    
    If given target is empty (target.size(0) == 0), the function will return 1.
    """
    with torch.no_grad():
        if target.size(0) == 0:
            # Since there is no data the accuracy is perfect i.e. 1
            return 1
        if compute_sigmoid:
            output = torch.sigmoid(output)
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


def binary_accuracy_with_logits(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification
    The output is assumed to be logits and will be passed through a sigmoid function."""
    return binary_accuracy(output, target, compute_sigmoid=True)

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[float]:
    r"""Taken from `~tllib.utils.metric.accuracy`.
    
    Computes the accuracy over the k top predictions for the specified values of k
    
    .. note::
        If given target is empty (target.size(0) == 0), the function will return 1.

    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.

    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        if target.size(0) == 0:
            # Since there is no data the accuracy is perfect i.e. 1
            return 1
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res