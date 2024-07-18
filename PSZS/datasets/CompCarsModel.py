from typing import Callable, Dict, List, Optional, Tuple, Any

from ._CompCarsClasses import *
from .CompCars import CompCars, DatasetDescriptor

class CompCarsModel(CompCars):
    multi_label = False
    def __init__(self, 
                 root: str, 
                 task: str, 
                 main_class_idx: int = 1, 
                 split: Optional[int] = None, 
                 descriptor: Optional[DatasetDescriptor] = None,
                 descriptor_file: Optional[str] = None,
                 transform: Optional[Callable] = None, 
                 annfile_dir: str = 'annfiles',
                 phase: Optional[str] = None,
                 ) -> None:
        super(CompCarsModel, self).__init__(root=root, 
                                            task=task, 
                                            main_class_idx=main_class_idx, 
                                            split=split, 
                                            descriptor=descriptor,
                                            descriptor_file=descriptor_file,
                                            transform=transform,
                                            annfile_dir=annfile_dir,
                                            phase=phase,
                                            )
        # # Only keep the main class
        # self.class_idx_name_maps = self.class_idx_name_maps[self.main_class_idx]
        # self.class_pred_idx_maps = self.class_pred_idx_maps[self.main_class_idx]
        # self.true_class_to_idx_maps = self.true_class_to_idx_maps[self.main_class_idx]
        # self.classes = self.classes[self.main_class_idx]
        
    @property
    def id_to_name(self) -> Dict[int, str]:
        """Dictionary that map the class id to the class name."""
        return self.descriptor.id_to_name[self.main_class_idx]
    
    @property
    def predIndex_to_targetId(self) -> Dict[int, int]:
        """Sequence of dictionaries that map the index of the prediction node to the 
        index of the class in the annotation file.
        This allows to recover the correct class for a prediction"""
        return self.descriptor.predIndex_to_targetId[self.main_class_idx]
    
    @property
    def targetId_to_predIndex(self) -> Dict[int, int]:
        """Dictionary that map the index of the class node to the index of the prediction node.
        Inverse of predIndex_to_targetId.
        This is required to map the ground truth to the corresponding internal class index
        which is required to compute the loss correctly."""
        return self.descriptor.targetId_to_predIndex[self.main_class_idx]
    
    @property
    def classes(self) -> List[str]:
        """Lists that contain the class names for only the model class."""
        return self.descriptor.classes[self.main_class_idx]
    @classes.setter
    def classes(self, value: List[str] | List[List[str]]):
        if isinstance(value[0], str):
            self.descriptor.classes[self.main_class_idx] = value
        elif isinstance(value[0], list):
            self.descriptor.classes = value
        else:
            raise ValueError(f"Invalid value type: ({(type(value))}) for classes")
            
    @property
    def eval_classes(self) -> List[int]:
        """List of class indices of the main level used for evaluation."""
        return list(self.descriptor.id_to_name[self.main_class_idx].keys())
    
    @property
    def num_classes(self) -> List[int]:
        """Number of classes for model level"""
        return self.descriptor.num_classes[self.main_class_idx]
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """Gets the next item in the dataset.
        Loads the image and target class index. The returned target is the index of the 
        class in the annotation file i.e. before any mappings.
        The target index is only for the main level i.e. the model.
        
        Args:
            index (int): Index of the sample
        
        Returns:
            Tuple[Any, int]: Tuple of image and target class index
        """
        img, target = super().__getitem__(index)
        return img, target[self.main_class_idx]
    