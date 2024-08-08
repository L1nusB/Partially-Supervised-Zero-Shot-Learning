
from typing import Dict, List, Tuple, Optional, Sequence, overload
import warnings

import numpy as np

import torch
import torch.nn.functional as F

F1_RETURN_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[Tuple[torch.Tensor,...],Tuple[torch.Tensor,...],Tuple[torch.Tensor,...]]
MULTIPLE_F1_RETURN_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]

def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value

def _format_label(label, is_indices, num_classes):
            """format various label to torch.Tensor."""
            if isinstance(label, np.ndarray):
                assert label.ndim == 2, 'The shape `pred` and `target` ' \
                    'array must be (N, num_classes).'
                label = torch.from_numpy(label)
            elif isinstance(label, torch.Tensor):
                if is_indices:
                    assert num_classes is not None, 'For index-type labels, ' \
                        'please specify `num_classes`.'
                    label = torch.stack([
                        label_to_onehot(indices, num_classes)
                        for indices in label
                    ])
                else:
                    assert label.ndim == 2, 'The shape `pred` and `target` ' \
                        'tensor must be (N, num_classes).'
            elif isinstance(label, Sequence):
                if is_indices:
                    assert num_classes is not None, 'For index-type labels, ' \
                        'please specify `num_classes`.'
                    label = torch.stack([
                        label_to_onehot(indices, num_classes)
                        for indices in label
                    ])
                else:
                    label = torch.stack(
                        [to_tensor(onehot) for onehot in label])
            else:
                raise TypeError(
                    'The `pred` and `target` must be type of torch.tensor or '
                    f'np.ndarray or sequence but get {type(label)}.')
            return label

def label_to_onehot(label: torch.Tensor|np.ndarray|Sequence|int, num_classes: int):
    """Convert a label to onehot format tensor.
    
    Taken from MMPretrain.

    Args:
        label (torch.Tensor|np.ndarray|Sequence|int): Label value.
        num_classes (int): The number of classes.

    Returns:
        torch.Tensor: The onehot format label tensor.

    Examples:
        >>> import torch
        >>> from mmpretrain.structures import label_to_onehot
        >>> # Single-label
        >>> label_to_onehot(1, num_classes=5)
        tensor([0, 1, 0, 0, 0])
        >>> # Multi-label
        >>> label_to_onehot([0, 2, 3], num_classes=5)
        tensor([1, 0, 1, 1, 0])
    """
    def format_to_tensor(value: torch.Tensor|np.ndarray|Sequence|int, 
                 device: Optional[torch.device]=None) -> torch.Tensor:
        """Convert various python types to label-format tensor.
        
        Taken from MMPretrain.

        Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
        :class:`Sequence`, :class:`int`.

        Args:
            value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.

        Returns:
            :obj:`torch.Tensor`: The foramtted label tensor.
        """
        if isinstance(value, torch.Tensor):
            device = value.device if device is None else device
        # Handle single number
        if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
            value = int(value.item())

        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value).to(torch.long)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            value = torch.tensor(value).to(torch.long)
        elif isinstance(value, int):
            value = torch.LongTensor([value])
        elif not isinstance(value, torch.Tensor):
            raise TypeError(f'Type {type(value)} is not an available label type.')
        assert value.ndim == 1, \
            f'The dims of value should be 1, but got {value.ndim}.'
        if device is not None:
            return value.to(device)
        else:
            return value
        
    label = format_to_tensor(label)
    sparse_onehot = F.one_hot(label, num_classes)
    return sparse_onehot.sum(0)

@torch.no_grad()
def accuracy(prediction: torch.Tensor, 
             target: torch.Tensor, 
             topk: Sequence[int]=(1,5), 
             originalTarget: Optional[torch.Tensor]=None,
             evalClasses: Optional[Sequence[int]] = None
             ) -> Tuple[List[torch.Tensor], int]:
    """Computes the prediction accuracy over the specified topk values.
    Via `evalClasses` only a subset of classes can be evaluated. Using `originalTarget`
    the original class labels can be specified which are referenced in `evalClasses`.
    This is relevant if `target` does not correspond to the original class labels 
    e.g. if the target indices are a (remapped) subset of the original classes and thus are not a 
    continuous range that could cause wrong classes to be evaluated.
    The number of relevant classes that were used for computing the accuracy is returned.

    Args:
        prediction (torch.Tensor): 
            Logit predictions of the model. Shape: (N, num_classes)
        target (torch.Tensor): 
            Ground truth target labels/indices based on num_classes of the model. Shape: (N,)
        topk (Sequence[int], optional): 
            Values to compute the accuracy over. Defaults to (1,5).
        originalTarget (Optional[Sequence[int]], optional): 
            Ground truth target labels based on the original dataset/annfile. Shape: (N,). Defaults to None.
        evalClasses (Optional[Sequence[int]], optional): 
            Target classes to be evaluated over. Should correspond to the indices in `target` or 
             classes from `originalTarget`. Defaults to None.

    Returns:
        Tuple[List[torch.Tensor], int]: List of accuracy values for each topk value and the number \
            of relevant classes.
    """
    maxk = min(max(topk), prediction.size()[1])
    _, pred = prediction.topk(maxk, 1, True, True)
    pred = pred.t()
    if evalClasses is None:
        # If no classes are specified all classes are evaluated
        # then originalClasses is irrelevant
        relevantTargetMask = torch.ones_like(target, dtype=bool)
    else:
        if originalTarget is None:
            warnings.warn('originalTarget is not specified when evalClasses given. '
                            'Use target directly instead. '
                            'This can cause unexpected results if target index does not '
                            'correspond to the original class index.')
            originalTarget = target
        # .cpu() is achieves significant speedup
        # .detach() is not necessary as we are in no_grad anyways and does not impact performance significantly
        # but might as well
        relevantTargetMask = torch.tensor([i in evalClasses for i in originalTarget.detach().cpu()])
    # Required to only account for relevant classes when building statistics
    num_relevant = relevantTargetMask.sum().item()
    # If no relevant classes are present return 100% accuracy
    # and avoid division by zero causing NaN
    if num_relevant == 0:
        return [torch.tensor(1)]*len(topk), 0
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    relevant_correct = correct[:,relevantTargetMask]
    return [relevant_correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / num_relevant for k in topk], num_relevant

@torch.no_grad()
def accuracy_hierarchy(prediction: torch.Tensor, 
                      target: torch.Tensor, 
                      hierarchy_map: Sequence[Dict[int, int]],
                      topk: int | Sequence[int]=(1,5), 
                      originalTarget: Optional[torch.Tensor]=None,
                      evalClasses: Optional[Sequence[int]] = None
                      ) -> Tuple[List[torch.Tensor], int]:...
@torch.no_grad()
def accuracy_hierarchy(prediction: torch.Tensor, 
                      target: torch.Tensor, 
                      hierarchy_map: Sequence[Dict[int, int]],
                      topk: int | Sequence[int]=(1,5), 
                      originalTarget: Optional[torch.Tensor]=None,
                      evalClasses: Optional[Sequence[Sequence[int]]] = None
                      ) -> Tuple[List[List[torch.Tensor]], List[int]]:...
@torch.no_grad()
def accuracy_hierarchy(prediction: torch.Tensor, 
                      target: torch.Tensor, 
                      hierarchy_map: Sequence[Dict[int, int]],
                      topk: int | Sequence[int]=(1,5), 
                      originalTarget: Optional[torch.Tensor]=None,
                      evalClasses: Optional[Sequence[int] | Sequence[Sequence[int]]] = None
                      ) -> Tuple[List[List[torch.Tensor | List[torch.Tensor]]], int | List[int]]:
    """Computes the prediction accuracy over the specified topk values for each level of the hierarchy
    as well as each collection of classes specified in `evalClasses`.
    The `target` is mapped using the `hierarchy_map` and the accuracy is computed for the `topk` values.
    Via `evalClasses` only a subset of classes can be evaluated. Using `originalTarget`
    the original class labels can be specified which are referenced in `evalClasses`.
    This is relevant if `target` does not correspond to the original class labels 
    e.g. if the target indices are a (remapped) subset of the original classes and thus are not a 
    continuous range that could cause wrong classes to be evaluated.
    Multiple class collections can be specified in `evalClasses` to evaluate the accuracy over multiple sets of classes.
    For each set of classes the accuracy and number of relevant classes is returned.
    If none or only a single set of classes is specified the return values are not nested.
    Accuracies are returned in the format: 
    
    `[[[eval1_lvl1_topk1, eval1_lvl1_topk2,...], [eval1_lvl2_topk1, eval1_lvl2_topk2,...],...], 
    [[eval2_lvl1_topk1, eval2_lvl1_topk2,...], [eval2_lvl2_topk1, eval2_lvl2_topk2,...],...]...]`

    Args:
        prediction (torch.Tensor): 
            Logit predictions of the model. Shape: (N, num_classes)
        target (torch.Tensor): 
            Ground truth target labels/indices based on num_classes of the model. Shape: (N,)
        hierarchy_map (Sequence[Dict[int, int]]):
            List of dictionaries that map the target classes (and predictions) to the hierarchy levels.
        topk (Sequence[int], optional): 
            Values to compute the accuracy over. Defaults to (1,5).
        originalTarget (Optional[Sequence[int]], optional): 
            Ground truth target labels based on the original dataset/annfile. Shape: (N,). Defaults to None.
        evalClasses (Optional[Sequence[int] | Sequence[Sequence[int]]], optional): 
            Target classes to be evaluated over. Should correspond to the indices in `target` or 
            classes from `originalTarget`. Defaults to None.

    Returns:
        Tuple[List[List[torch.Tensor] | List[List[torch.Tensor]]], int | List[int]]: 
            List of accuracy values for each topk value of each hierarchy level and the number of relevant classes.
            The first list corresponds to the hierarchy levels while the second list corresponds to the topk values.
    """
    if isinstance(topk, int):
        topk = (topk,)
    maxk = min(max(topk), prediction.size()[1])
    maxk = min(max(topk), prediction.size(1))
    
    if evalClasses is None:
        # If no classes are specified all classes are evaluated
        # then originalClasses is irrelevant
        # Convert to list for uniform handling below
        relevantTargetMasks = [torch.ones_like(target, dtype=bool)]
    else:
        if originalTarget is None:
            warnings.warn('originalTarget is not specified when evalClasses given. '
                            'Use target directly instead. '
                            'This can cause unexpected results if target index does not '
                            'correspond to the original class index.')
            originalTarget = target
        # .cpu() is achieves significant speedup
        # .detach() is not necessary as we are in no_grad anyways and does not impact performance significantly
        # but might as well
        if isinstance(evalClasses[0], int):
            # Convert to list for uniform handling below
            relevantTargetMasks = [torch.tensor([i in evalClasses for i in originalTarget.detach().cpu()])]
        else:
            # Make .cpu() and .detach() ONLY once outside loop
            o_trg = originalTarget.detach().cpu()
            relevantTargetMasks = [torch.tensor([i in eClasses for i in o_trg]) for eClasses in evalClasses]
    # Required to only account for relevant classes when building statistics
    num_relevants = [targetMask.sum().item() for targetMask in relevantTargetMasks]
    
    # len(relevantTargetMasks)xlen(topk)xlen(hierarchy_map)
    final_accs = torch.empty(len(relevantTargetMasks), len(topk), len(hierarchy_map), dtype=torch.float32)
    for lvl, lvl_map in enumerate(hierarchy_map):
        mapped_target = torch.tensor([lvl_map[t.item()] for t in target], dtype=target.dtype, device=target.device)
        _, pred = prediction.topk(maxk, 1, True, True)
        mapped_pred = torch.tensor([lvl_map[p.item()] for p in pred.flatten()], dtype=pred.dtype, device=pred.device).reshape(pred.size())
        mapped_pred = mapped_pred.t()
        
        for eval_group, (num_relevant, targetMask) in enumerate(zip(num_relevants, relevantTargetMasks)):
            # If no relevant classes are present return 100% accuracy
            # and avoid division by zero causing NaN
            if num_relevant == 0:
                final_accs[eval_group,:,lvl] = torch.ones(len(topk))
                continue
            correct = mapped_pred.eq(mapped_target.reshape(1, -1).expand_as(mapped_pred))
            relevant_correct = correct[:,targetMask]
            # Need to clamp values of each prediction to 1 to avoid multiple correct predictions being counted
            # possible for lower hierarchy levels (e.g. multiple models of the same make)
            final_accs[eval_group,:,lvl] = torch.Tensor([relevant_correct[:min(k, maxk)].float().sum(0).clamp(max=1).sum() * 100. / num_relevant for k in topk])
            
    # If only a single evalClasses was given return individual values instead of list
    # using len instead of isinstance for slightly better performance
    if len(num_relevants) == 1:
        return final_accs[0] , num_relevants[0]
    else:
        return final_accs, num_relevants

class PrecisionRecallF1:
    def __init__(self, 
                 num_classes: int,
                 device: torch.device,
                 topk: Optional[Sequence[int]] = None,
                 threshold: Optional[float] = None, 
                 evalClasses: Optional[Sequence[int]] = None,
                 average: str = 'macro'):
        # Use top-1 if nothing specified
        if threshold is None and topk is None:
            topk = 1
            
        if isinstance(topk, int):
            maxk = topk
        else:
            maxk = max(topk)
            
        
        self.num_classes = num_classes
        self.device = device
        self.topk = topk
        self.threshold = threshold
        self.maxk = maxk
        self.evalClasses = evalClasses
        if evalClasses is None:
            self.evalClassMask = torch.ones(num_classes, dtype=bool)
        else:
            self.evalClassMask = torch.tensor([i in evalClasses for i in range(num_classes)], dtype=bool)
        self.average = average
        
        if self.threshold is not None or isinstance(topk, int):
            self.update = self._update_single
        else:
            self.update = self._update_multiple
            
        self.reset()

    def reset(self):
        if self.threshold is not None or isinstance(self.topk, int):
            self.tp_sum = torch.zeros(self.num_classes, dtype=int, device=self.device)
            self.pred_sum = torch.zeros(self.num_classes, dtype=int, device=self.device)
        else:
            self.tp_sum = torch.zeros(len(self.topk), self.num_classes, dtype=int, device=self.device)
            self.pred_sum = torch.zeros(len(self.topk), self.num_classes, dtype=int, device=self.device)
        self.gt_sum = torch.zeros(self.num_classes, dtype=int, device=self.device)
    
    @torch.no_grad()   
    def _update_single(self, 
                       prediction: torch.Tensor, 
                       target: torch.Tensor,) -> None:
        pred_class_prob = _format_label(prediction, False, self.num_classes)
        target_one_hot = _format_label(target, True, self.num_classes)
        self.gt_sum += target_one_hot.sum(0)
        
        if self.threshold is not None:
            pos_inds = (pred_class_prob >= self.threshold).long()
        else:
            _, topk_indices = pred_class_prob.topk(self.maxk, 1)
            pos_inds = torch.zeros_like(pred_class_prob, dtype=int).scatter_(1, topk_indices, 1)
            
        class_correct = pos_inds & target_one_hot
        self.tp_sum += class_correct.sum(0)
        self.pred_sum += pos_inds.sum(0)
    
    @torch.no_grad()       
    def _update_multiple(self, 
                         prediction: torch.Tensor, 
                         target: torch.Tensor,) -> None:
        pred_class_prob = _format_label(prediction, False, self.num_classes)
        target_one_hot = _format_label(target, True, self.num_classes)
        self.gt_sum += target_one_hot.sum(0)
        
        # If multiple topk construct metrics for each k
        _, topk_indices = pred_class_prob.topk(self.maxk, 1)
        for i,k in enumerate(self.topk):
            pos_inds = torch.zeros_like(pred_class_prob, dtype=int).scatter_(1, topk_indices[:,:k], 1)
            
            class_correct = pos_inds & target_one_hot
            self.tp_sum[i] += class_correct.sum(0)
            self.pred_sum[i] += pos_inds.sum(0)

    # Set to _update_single or _update_multiple
    @torch.no_grad()   
    def update(self, 
               prediction: torch.Tensor, 
               target: torch.Tensor,) -> None:
        """Computes and stores/updates the number of true positive, predicted positive 
        and ground truth positive.
        Can be used for both single and multiple topk values.
        The respective function is set to override this method in the constructor.

        Args:
            prediction (torch.Tensor): 
                Unnormalized logits of the prediction. Shape: (N, num_classes)
            target (torch.Tensor): 
                Class indices of the prediction target. Shape: (N,)
        """
        raise NotImplementedError('This method should be set to _udpate_single or _update_multiple.')

    @overload
    @torch.no_grad()   
    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:...
    
    @overload
    @torch.no_grad()   
    def compute(self) -> Tuple[Tuple[torch.Tensor,...],Tuple[torch.Tensor,...],Tuple[torch.Tensor,...]]:...
    
    @torch.no_grad()   
    def compute(self) -> F1_RETURN_TYPE:
        if self.threshold is not None or isinstance(self.topk, int):
            precision = self.tp_sum / torch.clamp(self.pred_sum, min=1).float() * 100
            recall = self.tp_sum / torch.clamp(self.gt_sum, min=1).float() * 100
            f1_score = 2 * precision * recall / torch.clamp(precision + recall, min=torch.finfo(torch.float32).eps)
            if self.average == 'macro':
                precision = precision[self.evalClassMask].mean(0)
                recall = recall[self.evalClassMask].mean(0)
                f1_score = f1_score[self.evalClassMask].mean(0)
        else:
            precision = ()
            recall = ()
            f1_score = ()
            for i in range(len(self.topk)):
                precision_k = self.tp_sum[i] / torch.clamp(self.pred_sum[i], min=1).float() * 100
                recall_k = self.tp_sum[i] / torch.clamp(self.gt_sum, min=1).float() * 100
                f1_score_k = 2 * precision_k * recall_k / torch.clamp(precision_k + recall_k, min=torch.finfo(torch.float32).eps)
                if self.average == 'macro':
                    precision += (precision_k[self.evalClassMask].mean(0),)
                    recall += (recall_k[self.evalClassMask].mean(0),)
                    f1_score += (f1_score_k[self.evalClassMask].mean(0),)
                else:
                    precision += (precision_k,)
                    recall += (recall_k,)
                    f1_score += (f1_score_k,)
        return precision, recall, f1_score

@overload
def prec_rec_f1(output: torch.Tensor, target: torch.Tensor,
                thr: Optional[float] = None, topk: Optional[int] = None,
                evalClasses: Optional[Sequence[int]] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

@overload 
def prec_rec_f1(output: torch.Tensor, target: torch.Tensor,
                thr: Optional[float] = None, topk: Sequence[int] = None,
                evalClasses: Optional[Sequence[int]] = None,
                ) -> Tuple[Tuple[torch.Tensor,...],Tuple[torch.Tensor,...],Tuple[torch.Tensor,...]]: ...
   
def prec_rec_f1(output: torch.Tensor, target: torch.Tensor, 
                thr: Optional[float] = None, topk: Optional[int|Sequence[int]] = None,
                evalClasses: Optional[Sequence[int]] = None,
                ) -> F1_RETURN_TYPE:
    """Calculate base classification task metrics, such as  precision, recall,
    f1_score, support. 
    
    Taken partially from MMPretrain with average='macro' configuration and without support.

    Args:
        output (torch.Tensor): Output predictions of model. 
                            (Assumed to be probabilities but indices should work as well)
                            Shape: (N, num_classes)
        target (torch.Tensor): Ground truth target labels as the class index. Shape: (N,)
        thr (float, optional): Predictions with scores under the thresholds are considered as negative. (default: None)
        topk (int, Sequence[int], optional): Predictions with the k-th highest scores are \
                considered as positive. Defaults to None. If Sequence is given Results for each \
                component are returned.
        evalClasses (Sequence[int], optional): Class indices to calculate metrics for. \
            If None all classes are considered. (default: None)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Precision, Recall, F1 Score
        If topk is a Sequence each element becomes a Tuple[torch.Tensor]
        
    Notes:
            If both ``thr`` and ``topk`` are set, use ``topk` to determine
            positive predictions. If neither is set, use ``topk=1`` as
            default.
    """
    with torch.no_grad():
        precision = recall = f1_score = 0
        num_classes = output.size(1)
        
        pred_class_prob = _format_label(output, False, num_classes)
        target_one_hot = _format_label(target, True, num_classes)
        
        # Use top-1 if nothing specified
        if thr is None and topk is None:
            topk = 1
            
        if isinstance(topk, int):
            maxk = topk
        else:
            maxk = max(topk)
            
        if thr is not None:
            pos_inds = (pred_class_prob >= thr).long()
        else:
            _, topk_indices = pred_class_prob.topk(maxk, 1)
            pos_inds = torch.zeros_like(pred_class_prob, dtype=int).scatter_(1, topk_indices, 1)
            
        if evalClasses is None:
            classMask = torch.ones(target_one_hot.size(1), dtype=bool)
        else:
            classMask = torch.zeros(target_one_hot.size(1), dtype=bool)
            classMask[evalClasses] = True
            
        if isinstance(topk, int):
            class_correct = pos_inds & target_one_hot
            
            cc_sum = class_correct.sum(0)

            precision = cc_sum / torch.clamp(pos_inds.sum(0), min=1).float() * 100
            recall = cc_sum / torch.clamp(target_one_hot.sum(0), min=1).float() * 100
            f1_score = 2 * precision * recall / torch.clamp(precision + recall, min=torch.finfo(torch.float32).eps)
            precision = precision[classMask].mean(0)
            recall = recall[classMask].mean(0)
            f1_score = f1_score[classMask].mean(0)
        else:
            # If multiple topk construct metrics for each k
            precision = ()
            recall = ()
            f1_score = ()
            for k in topk:
                pos_inds = torch.zeros_like(pred_class_prob, dtype=int).scatter_(1, topk_indices[:,:k], 1)
                
                class_correct = pos_inds & target_one_hot
            
                cc_sum = class_correct.sum(0)
                precision_k = cc_sum / torch.clamp(pos_inds.sum(0), min=1).float() * 100
                recall_k = cc_sum / torch.clamp(target_one_hot.sum(0), min=1).float() * 100
                f1_score_k = 2 * precision_k * recall_k / torch.clamp(precision_k + recall_k, min=torch.finfo(torch.float32).eps)
                # Only consider classes that are evaluated
                precision += (precision_k[classMask].mean(0),)
                recall += (recall_k[classMask].mean(0),)
                f1_score += (f1_score_k[classMask].mean(0),)

        return precision, recall, f1_score
    

class MultiplePrecisionRecallF1:
    """Precision, Recall and F1 Score metric for multiple eval Classes.
    
    Taken partially from MMPretrain with average='macro' configuration and without support.
    """
    def __init__(self, 
                 num_classes: int | Sequence[int],
                 device: torch.device,
                 topk: Optional[int] = None,
                 threshold: Optional[float] = None, 
                 evalClasses: Optional[Sequence[Sequence[int]]] = None,
                 average: str = 'macro'):
        # Use top-1 if nothing specified
        if threshold is None and topk is None:
            topk = 1
        
        self.num_classes = num_classes
        self.device = device
        self.topk = topk
        self.threshold = threshold
        self.evalClasses = evalClasses
        self.average = average
        if evalClasses is None:
            self.evalClassMask = torch.ones(1, num_classes, dtype=bool)
        else:
            self.evalClassMask = torch.tensor([[i in eClasses for i in range(num_classes)] for eClasses in evalClasses], 
                                              dtype=bool)
        
        if self.threshold is None:
            self._compute_pos_inds = self._compute_pos_inds_topk
        else:
            self._compute_pos_inds = self._compute_pos_inds_thres
            
        self.reset()

    def reset(self):
        self.tp_sum: torch.Tensor = torch.zeros(self.num_classes, dtype=int, device=self.device)
        self.pred_sum: torch.Tensor = torch.zeros(self.num_classes, dtype=int, device=self.device)
        self.gt_sum: torch.Tensor = torch.zeros(self.num_classes, dtype=int, device=self.device)
        
    def _compute_pos_inds_thres(self, pred_cls_prob: torch.Tensor) -> torch.Tensor:
        return (pred_cls_prob >= self.threshold).long()
    
    def _compute_pos_inds_topk(self, pred_cls_prob: torch.Tensor) -> torch.Tensor:
        _, topk_indices = pred_cls_prob.topk(self.topk, 1)
        return torch.zeros_like(pred_cls_prob, dtype=int).scatter_(1, topk_indices, 1)
    
    @torch.no_grad()   
    def update(self, 
               prediction: torch.Tensor, 
               target: torch.Tensor,) -> None:
        """Computes and stores/updates the number of true positive, predicted positive 
        and ground truth positive.

        Args:
            prediction (torch.Tensor): 
                Unnormalized logits of the prediction. Shape: (N, num_classes)
            target (torch.Tensor): 
                Class indices of the prediction target. Shape: (N,)
        """
        pred_class_prob = _format_label(prediction, False, self.num_classes)
        target_one_hot = _format_label(target, True, self.num_classes)
        self.gt_sum += target_one_hot.sum(0)
        
        if self.threshold is not None:
            pos_inds = (pred_class_prob >= self.threshold).long()
        else:
            _, topk_indices = pred_class_prob.topk(self.topk, 1)
            pos_inds = torch.zeros_like(pred_class_prob, dtype=int).scatter_(1, topk_indices, 1)
            
        class_correct = pos_inds & target_one_hot
        self.tp_sum += class_correct.sum(0)
        self.pred_sum += pos_inds.sum(0)

    @overload
    @torch.no_grad()   
    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:...
    @overload
    @torch.no_grad()   
    def compute(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:...
    
    @torch.no_grad()   
    def compute(self) -> MULTIPLE_F1_RETURN_TYPE:
        """Computes the precision, recall and f1 score based on the currently stored values.
        Depending on `self.average` either the micro metrics are returned or the macro metrics
        are computed for all eval class collections specified in `self.evalClasses`.

        Returns:
            MULTIPLE_F1_RETURN_TYPE: Precision, Recall, F1 Score based on the average configuration.
        """
        precision = self.tp_sum / torch.clamp(self.pred_sum, min=1).float() * 100
        recall = self.tp_sum / torch.clamp(self.gt_sum, min=1).float() * 100
        f1_score = 2 * precision * recall / torch.clamp(precision + recall, 
                                                        min=torch.finfo(torch.float32).eps)
        if self.average == 'macro':
            precision = [precision[evalMask].mean(0) for evalMask in self.evalClasses]
            recall = [recall[evalMask].mean(0) for evalMask in self.evalClasses]
            f1_score = [f1_score[evalMask].mean(0) for evalMask in self.evalClasses]
        return precision, recall, f1_score