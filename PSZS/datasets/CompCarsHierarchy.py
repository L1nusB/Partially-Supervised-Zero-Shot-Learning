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
    
    # def __getitem__(self, index: int) -> Tuple[Any, List[int]]:
    #     """
    #     Args:
    #         index (int): Index of the sample
    #         return (tuple): (image, target) where target is a list of indeces of the target classes for each level.
    #     """
    #     path, target, _ = self.samples[index]
    #     # Map the true class index to the internal class index
    #     target = [self.true_class_to_idx_maps[i][t] for i, t in enumerate(target)]
    #     img = self.loader(path)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     if self.target_transform is not None and target is not None:
    #         target = [self.target_transform(t) for t in target]
    #     return img, target
    
    # @property
    # def num_classes(self) -> List[int]:
    #     """Number of classes for each level"""
    #     return [len(cls) for cls in self.classes]