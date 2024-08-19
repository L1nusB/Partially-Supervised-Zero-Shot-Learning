from typing import Callable, Optional

from ._CompCarsClasses import *
from .CompCars import CompCars, DatasetDescriptor

CLASSES_LEVELS = [
    TYPE_CLASSES_SV_WEB,
    MAKE_CLASSES_SV_WEB,
]

class CompCarsHierarchy(CompCars):
    multi_label = True
    
    def __init__(self, 
                 root: str, 
                 task: str, 
                 main_class_idx: int = -1, 
                 split: Optional[int] = None, 
                 descriptor: Optional[DatasetDescriptor] = None,
                 descriptor_file: Optional[str] = None,
                 label_index: bool = False,
                 transform: Optional[Callable] = None, 
                 annfile_dir: str = 'annfiles',
                 phase: Optional[str] = None,
                 infer_all_classes : bool = True,
                 ) -> None:
        super(CompCarsHierarchy, self).__init__(root=root, 
                                                task=task, 
                                                main_class_idx=main_class_idx, 
                                                split=split, 
                                                descriptor=descriptor,
                                                descriptor_file=descriptor_file,
                                                label_index=label_index,
                                                transform=transform,
                                                annfile_dir=annfile_dir,
                                                phase=phase,
                                                )
                                            
        # Self.classes is set in super
        # If not infer_all_classes, use "all" classes of the higher granularity irrespective
        # of what actually exists in the dataset
        if infer_all_classes==False:
            # Append the more granual classes as well
            self.classes = CLASSES_LEVELS[-self.most_detailed_class:] + [self.classes[self.most_detailed_class]]