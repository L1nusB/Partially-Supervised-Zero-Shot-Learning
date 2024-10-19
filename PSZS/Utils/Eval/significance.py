from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy.stats import permutation_test
from scipy.stats._resampling import PermutationTestResult
import matplotlib.pyplot as plt

@torch.no_grad()
def get_pred_and_target(loader: DataLoader, 
                        model: nn.Module,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    preds = []
    targets = []
    for data, labels in loader:
        # a : torch.Tensor = model(data)
        # b = a.argmax(dim=1)
        preds.append(model(data).argmax(dim=1).cpu())
        targets.append(labels.cpu())
        
    preds = np.array(torch.cat(preds, dim=0))
    targets = np.array(torch.cat(targets, dim=0))
    return preds, targets

def calculate_scores(in1: Tuple[np.ndarray, np.ndarray],
                     in2: Tuple[np.ndarray, np.ndarray],
                     num_divisions=2,
                     num_k_validation=5,) -> Tuple[dict, dict]:
    preds1, targets1 = in1
    preds2, targets2 = in2
    index = np.arange(len(targets1),dtype=int)
    np.random.shuffle(index)
    fold_size = len(index) // num_divisions
    acc_1 , acc_2 = [] , []
    recall_1 , recall_2 = [] , [] 
    precision_1 , precision_2 = [] , []
    f1_score_1, f1_score_2 = [] , [] 
    
    for _ in range(num_k_validation):
        np.random.shuffle(index)
        for i in range(num_divisions):
            start = i * fold_size
            end = (i+1) * fold_size
            selected_index = index[start:end]
            
            acc_1.append(accuracy_score(targets1[selected_index],preds1[selected_index]))
            precision_1.append(precision_score(targets1[selected_index],preds1[selected_index],average='macro',zero_division=0))
            recall_1.append(recall_score(targets1[selected_index],preds1[selected_index],average='macro',zero_division=0))
            f1_score_1.append(f1_score(targets1[selected_index],preds1[selected_index],average='macro',zero_division=0))


        for i in range(num_divisions):
            start = i * fold_size
            end = (i+1) * fold_size
            selected_index = index[start:end]
            
            f1_score_2.append(f1_score(targets2[selected_index],preds2[selected_index],average='macro',zero_division=0))
            acc_2.append(accuracy_score(targets2[selected_index],preds2[selected_index]))
            recall_2.append(recall_score(targets2[selected_index],preds2[selected_index],average='macro',zero_division=0))
            precision_2.append(precision_score(targets2[selected_index],preds2[selected_index],average='macro',zero_division=0))

    scores1 = dict(accuracy_score=acc_1,precision_score=precision_1,recall_score=recall_1,f1_score=f1_score_1)
    scores2 = dict(accuracy_score=acc_2,precision_score=precision_2,recall_score=recall_2,f1_score=f1_score_2)

    return scores1, scores2

def _statistic(a , b):
    return (np.mean(a)- np.mean(b))

def compute_permutation_test(scores1: dict, scores2: dict) -> PermutationTestResult:
    for key in scores1.keys():
        res = permutation_test((scores1[key],scores2[key]),_statistic, permutation_type="independent",n_resamples=np.inf)
        print(f"-------------------------")
        print("> Metric:\t{}".format(key)) 
        print(f"> Score1\t{scores1[key]}")
        print(f"> Score2\t{scores2[key]}")
        print("> Statistic Percent:\t{}".format(res.statistic*100)) 
        print("> P-Value:\t{:e}".format(res.pvalue))
    return res

def plot(res: PermutationTestResult,
         outPath: str = "significance.png") -> None:
    fig, ax = plt.subplots()
    ax: plt.Axes
    a = ax.hist(res.null_distribution, bins=10, density=True)
    print(a)
    plt.savefig(outPath)
    
def signifance_test(loader: DataLoader,
                    model1: nn.Module,
                    model2: nn.Module,
                    num_divisions: int = 2,
                    num_k_validations: int = 5,
                    outFile: str = "significance.png") -> None:
    model1.eval()
    model2.eval()
    output1 = get_pred_and_target(loader, model1)
    output2 = get_pred_and_target(loader, model2)
    scores1, scores2 = calculate_scores(output1, output2, num_divisions=num_divisions, num_k_validation=num_k_validations)
    permutation_res = compute_permutation_test(scores1, scores2)
    plot(permutation_res, outPath=outFile)