import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

from .CustomDataset import CustomDataset, DatasetDescriptor

class CUB(CustomDataset):
    image_list = {
        'rawTot' : 'cubRaw_total.txt',
        'rawNov' : 'cubRaw_novel.txt',
        'rawShr' : 'cubRaw_shared.txt',
        'drawingTot' : 'cubDrawing_total.txt',
        'drawingNov' : 'cubDrawing_novel.txt',
        'drawingShr' : 'cubDrawing_shared.txt',
    }
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
                 ) -> None:
        assert task in self.image_list, f'Task {task} not included. Available: {",".join(self.image_list.keys())}'
        if split is not None:
            annfile_dir = f'{annfile_dir}_Split{split}'
            
        if phase is not None:
            # Add phase to the annfile name
            annfile_path = os.path.join(root, 
                                        annfile_dir, 
                                        self.image_list[task].replace('.txt', f'_{phase}.txt'))
        else:    
            annfile_path = os.path.join(root, 
                                        annfile_dir, 
                                        self.image_list[task])
        if descriptor_file is not None:
            descriptor_file = os.path.join(root, annfile_dir, descriptor_file)
        
        self.hierarchy_level_names = ['Order', 'Family', 'Genus', 'Species']
        # Omit many arguments that are just not used in practice (currently)
        super(CUB, self).__init__(root, 
                                  annfile_path=annfile_path,
                                  descriptor=descriptor,
                                  descriptor_file=descriptor_file,
                                  transform=transform)
        
        # The index in the target label corresponding to the main class
        # e.g., if the format is [classIndexType][sub_sep][classIndexMake][sub_sep][classIndexModel]
        # if the main label is the model then main_class_idx=2
        # if the main label is the make then main_class_idx=1
        # Use negative values as default to indicate the last class
        if main_class_idx < 0:
            main_class_idx = len(self.samples[0][1]) + main_class_idx
        self.main_class_idx = main_class_idx
        
        self.most_detailed_class = len(self.samples[0][1])-1
        if self.most_detailed_class != main_class_idx:
            print(f"Main class ({main_class_idx}) is not the most detailed class ({self.most_detailed_class}). "
                  f"Using specified main class ({main_class_idx}) as highest detail class.")
            self.most_detailed_class = main_class_idx
            
    @property
    def id_to_name(self) -> List[Dict[int, str]]:
        """List of dictionaries that map the class id to the class name."""
        return self.descriptor.id_to_name
    
    @property
    def predIndex_to_targetId(self) -> List[Dict[int, int]]:
        """Sequence of dictionaries that map the index of the prediction node to the 
        index of the class in the annotation file.
        This allows to recover the correct class for a prediction"""
        return self.descriptor.predIndex_to_targetId
    
    @property
    def targetId_to_predIndex(self) -> List[Dict[int, int]]:
        """List of dictionaries that map the index of the class node to the index of the prediction node.
        Each entry in the list corresponds to a level in the hierarchy.
        Inverse of predIndex_to_targetId.
        This is required to map the ground truth to the corresponding internal class index
        which is required to compute the loss correctly."""
        return self.descriptor.targetId_to_predIndex
    
    @property
    def classes(self) -> List[List[str]]:
        """List of lists that contain the class names for each level in the hierarchy."""
        return self.descriptor.classes
    @classes.setter
    def classes(self, value: List[List[str]]):
        self.descriptor.classes = value
            
    @property
    def eval_classes(self) -> List[int]:
        """List of class indices of the most detailed class level for evaluation."""
        return list(self.id_to_name[self.most_detailed_class].keys())
    
    @property
    def num_classes(self) -> List[int]:
        """Number of classes for each level"""
        return self.descriptor.num_classes
            
    def parse_data_file(self, file_name: str, 
                        main_sep: str = '=', 
                        sub_sep: str = "#") -> List[Tuple[str, List[int], List[str]]]:
        """Parse file to data list with custom CompCars data file Format
        Expected format:
        [relativePath]=[ClassIndices]=[ClassNames]

        Within [ClassIndices] and [ClassNames] multiple fields can be set by [sub_sep] separator. 

        Args:
            file_name (str): The path of data file
            main_sep (str): Main separator to split relativePath, ClassIndices and ClassNames. Defaults to '='
            sub_sep (str): Separator to split within ClassIndices and ClassNames. Defaults to '#'
            return (list): List of (image path, class_indices, class_names) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.read().splitlines():
                split_line = line.split(sep=main_sep)
                path = split_line[0]
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = [int(i) for i in split_line[1].split(sep=sub_sep)]
                names = [i for i in split_line[2].split(sep=sub_sep)]
                data_list.append((path, target, names))
        return data_list
    
    def __getitem__(self, index: int) -> Tuple[Any, int | Sequence[int]]:
        """Gets the next item in the dataset.
        Loads the image and target class index. The returned target is the index of the 
        class in the annotation file i.e. before any mappings.
        
        Args:
            index (int): Index of the sample
        
        Returns:
            Tuple[Any, int | Sequence[int]]: Tuple of image and target class indices
        """
        path, target, _ = self.samples[index]
        # Map the true class index to the internal class index
        # target = self.true_class_to_idx_maps[self.main_class_idx][target[self.main_class_idx]]
        img = self.loader(path)
        # Target_transform are not used in practice
        # thus removed here for simplicity (+ speedup)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    