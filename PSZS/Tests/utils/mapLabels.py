from typing import List, Optional, Sequence, Tuple, overload
import torch

from PSZS.datasets import transform_target
from PSZS.datasets import DatasetDescriptor

__all__ = ['map_labels']

@overload
def map_labels(labels: torch.Tensor, 
                mode: str, 
                descriptor: DatasetDescriptor,
                hierarchy_level: Optional[int]=None,
                multiple: bool = False) -> torch.Tensor:...
    
@overload
def map_labels(labels: Tuple[torch.Tensor, ...], 
                mode: str, 
                descriptor: Sequence[DatasetDescriptor],
                hierarchy_level: Optional[Sequence[int]]=None,
                multiple: bool = True) -> List[torch.Tensor]:...

def map_labels(labels: torch.Tensor | Tuple[torch.Tensor, ...], 
                mode: str, 
                descriptor: DatasetDescriptor | Sequence[DatasetDescriptor],
                hierarchy_level: Optional[int | Sequence[int]]=None,
                multiple: bool = False) -> torch.Tensor | List[torch.Tensor]:
    if multiple:
        return _map_labels_multiple(labels, mode, descriptor, hierarchy_level)
    else:
        return _map_labels_single(labels, mode, descriptor, hierarchy_level)
          
def _map_labels_multiple(labels: Tuple[torch.Tensor, ...], 
                         mode: str, 
                         descriptors: DatasetDescriptor | Sequence[DatasetDescriptor],
                         hierarchy_levels: Optional[int | Sequence[int]]=None) -> List[torch.Tensor]:
        """Map the (ground truth) labels to based on the dataset descriptors and the specified `mode`.
        Mapping is performed based on `self.hierarchy_levels` or `self.hierarchy_level_val` of the used datasets.
        Available values for `mode` are based on `~PSZS.datasets.transform_target()`
            - 'target': Map the internal prediction index to the target class as in the annotation file
            - 'pred': Map the (original) target class to the internal prediction index
            - 'name': Map the target class to the class name. Needs to be the original target class, not the internal prediction index.
            - 'targetIndex': Map the class id to the index of that class in the target (independent of internal prediction)

        Args:
            labels (Tuple[torch.Tensor, ...]): Labels to be mapped
            mode (str): mode of the mapping is reversed.
            descriptors (DatasetDescriptor | Sequence[DatasetDescriptor]): Dataset descriptors for the labels. 
            Gets repeated if only one is provided.
            hierarchy_levels (Optional[int | Sequence[int]], optional): Hierarchy levels for the labels. 
            Gets repeated if only one is provided. Defaults to None.

        Returns:
            List[torch.Tensor]: Mapped ground truth labels.
        """
        if isinstance(descriptors, DatasetDescriptor):
            descriptors = [descriptors] * len(labels)
        if hierarchy_levels is None:
            hierarchy_levels = [None] * len(labels)
        elif isinstance(hierarchy_levels, int):
            hierarchy_levels = [hierarchy_levels] * len(labels)
        return [transform_target(target=gt, mode=mode, descriptor=desc, hierarchy_level=lvl) 
                        for gt, desc, lvl in zip(labels, descriptors, hierarchy_levels)]
            
def _map_labels_single(labels: torch.Tensor, 
                        mode: str, 
                        descriptor: DatasetDescriptor,
                        hierarchy_level: Optional[int]=None) -> torch.Tensor:
        """Map the (ground truth) labels to based on the dataset descriptors and the specified `mode`.
        Mapping is performed based on `self.hierarchy_levels` or `self.hierarchy_level_val` of the used datasets.
        Available values for `mode` are based on `~PSZS.datasets.transform_target()`
            - 'target': Map the internal prediction index to the target class as in the annotation file
            - 'pred': Map the (original) target class to the internal prediction index
            - 'name': Map the target class to the class name. Needs to be the original target class, not the internal prediction index.
            - 'targetIndex': Map the class id to the index of that class in the target (independent of internal prediction)

        Args:
            labels (torch.Tensor): Labels to be mapped
            mode (str): mode of the mapping is reversed.

        Returns:
            torch.Tensor: Mapped ground truth labels.
        """
        return transform_target(target=labels, mode=mode, descriptor=descriptor, 
                                    hierarchy_level=hierarchy_level)