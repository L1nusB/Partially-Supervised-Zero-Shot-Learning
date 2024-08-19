from collections import defaultdict
import os
import inspect
from typing import Dict, Optional, Callable, Sequence, Set, Tuple, Any, List

import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader

class DatasetDescriptor:
    # Sequence of dictionaries that map the class id to the class name. 
    # Each entry in the list corresponds to a level in the hierarchy.
    id_to_name: List[Dict[int, str]] = [{}]
    
    # Sequence of dictionaries that map the class id to a continuous index starting at 0 to num_classes. 
    # This index is independent of the internal prediction and its indices and offsets.
    # Each entry in the list corresponds to a level in the hierarchy.
    # If no offset is given this is the same as targetId_to_predIndex
    id_to_index: List[Dict[int, int]] = [{}]
    
    # Sequence of dictionaries that map the index of the prediction node to the index of the class 
    # in the annotation file.
    # Each entry in the list corresponds to a level in the hierarchy
    # This is the inverse of targetId_to_predIndex
    # A given offset to the prediction index can be specified.
    # E.g. consider that the dataset contains the classes [1,3,4,7,12] we would have the following mapping
    # {0:1, 1:3, 2:4, 3:7, 4:12}
    # This allows to recover the correct class for a prediction e.g. if the model predicts for index 2
    # this will correspond to class 4 etc.
    predIndex_to_targetId: List[Dict[int, int]] = [{}]
    
    # Sequence of dictionaries that map the index of the class node to the index of the prediction node.
    # Each entry in the list corresponds to a level in the hierarchy.
    # This is the inverse of predIndex_to_targetId
    # A given offset to the prediction index can be specified.
    # E.g. for the same example as above we would have the following mapping
    # {1:0, 3:1, 4:2, 7:3, 12:4}
    # This is required to map the ground truth to the corresponding internal class index
    # which is required to compute the loss correctly.
    # E.g. if we have the prediction for index 2 the correct ground truth class would be 4
    # but we don't want to use class index 4 but the internal index 2 as the target as this correponds
    # to the 3rd prediction node which is the one for class 4.
    # If no offset is given this is the same as id_to_index
    targetId_to_predIndex: List[Dict[int, int]] = [{}]
    
    # Map from the coarse to the fine level in the hierarchy.
    # The fine label is given as the last index in the list of indices.
    # This pertains to the absolute/true labels and not the prediction indices.
    coarse_fine_map: List[Dict[int, int]] = [{}]
    # Map from the fine level to the coarse level in the hierarchy.
    # The fine label is given as the last index in the list of indices.
    # This pertains to the absolute/true labels and not the prediction indices.
    # Inverse of coarse_fine_map
    fine_coarse_map: List[Dict[int, int]] = [{}]
    
    # Map from the coarse to the fine level in the hierarchy.
    # The fine label is given as the last index in the list of indices.
    # This pertains to the prediction indices as compared to coarse_fine_map.
    pred_coarse_fine_map: List[Dict[int, int]] = [{}]
    # Map from the fine level to the coarse level in the hierarchy.
    # The fine label is given as the last index in the list of indices.
    # This pertains to the prediction indices as compared to fine_coarse_map.
    # Inverse of pred_coarse_fine_map
    pred_fine_coarse_map: List[Dict[int, int]] = [{}]
    
    # Map from the fine level to the coarse level in the hierarchy with the names being the values.
    # Internally a combination of pred_fine_coarse_map and classes is used to construct this.
    fine_coarse_name_map: List[Dict[int, int]] = [{}]
    
    # Sequence of lists that contain the class names for each level in the hierarchy.
    # Each entry in the list corresponds to a level in the hierarchy.
    # The lists are ordered by the prediction index i.e. the first class corresponds to the first prediction node etc.
    classes : List[List[str]] = [[]]
    
    hierarchy_level_names: List[str] = []
    
    def __init__(self, 
                 filePath: str,
                 hierarchy_levels: Optional[int] = None,
                 offset: Optional[List[int]] = None,
                 level_names: Optional[List[str]] = None,
                 ) -> None:
        self.default_names = True
        data = self.parse_data_file(filePath)
        
        if hierarchy_levels is None:
            hierarchy_levels = len(data[0][0])
        # Needs to be set before setting level_names as property references it
        self.hierarchy_levels = hierarchy_levels
        
        if offset is None:
            offset = [0]*hierarchy_levels
        if level_names is None:
            # Directly set private variable to avoid property setter
            # that sets default_names to False
            self._hierarchy_level_names = [f"Level {i}" for i in range(hierarchy_levels)]
        else:
            assert len(level_names) == hierarchy_levels, f"Number of level names ({len(level_names)}) must match hierarchy levels ({hierarchy_levels})."
            # Use property setter to set default_names to False
            self.hierarchy_level_names = level_names
            
        self.offset = offset
        self.coarse_fine_map = [{indices[i]:indices[-1] for indices, _ in data} for i in range(hierarchy_levels)]
        self.fine_coarse_map = [{indices[-1]:indices[i] for indices, _ in data} for i in range(hierarchy_levels)]
        self.id_to_name = [{indices[i]:names[i] for indices, names in data} for i in range(hierarchy_levels)]
        # Need to sort the keys to remove potentially wrong order from descriptor file
        # Otherwise the indices will not match e.g. we map make 24 to index 0, 17 to index 1 etc.
        # This is because the description file is sorted by modelID for 
        # which we also do the sorting in the following dict
        self.id_to_index = [{classID:index for index, classID in enumerate(sorted(level.keys()))} for level in self.id_to_name]
        # Apply/Respect offset if specified
        self.predIndex_to_targetId = [{predIndex+offset[i]:targetID for predIndex, targetID in enumerate(sorted(list(level.keys())))}
                                        for i, level in enumerate(self.id_to_name)]
        self.targetId_to_predIndex = [{v:k for k,v in level.items()} for level in self.predIndex_to_targetId]
        
        self.pred_coarse_fine_map = [{self.targetId_to_predIndex[i][k]:self.targetId_to_predIndex[-1][v] for k,v in self.coarse_fine_map[i].items()} 
                                         for i in range(self.hierarchy_levels)]
        self.pred_fine_coarse_map = [{self.targetId_to_predIndex[-1][k]:self.targetId_to_predIndex[i][v] for k,v in self.fine_coarse_map[i].items()}
                                            for i in range(self.hierarchy_levels)]
        
        # Number of classes for each level
        # Duplicate makes are automatically removed due to the dictionary structure
        self.num_classes = [len(level) for level in self.id_to_name]
        
        # Respect level offset during indexing
        # Order of classes will be w.r.t. prediction index i.e. first class will correspond to 
        # first prediction node etc.
        self.classes = [[self.id_to_name[level][self.predIndex_to_targetId[level][class_idx+offset[level]]] 
                                for class_idx in range(self.num_classes[level])] 
                            for level in range(hierarchy_levels)]
        
        self.fine_coarse_name_map = [{model:self.classes[i][make] 
                                        for model, make in self.pred_fine_coarse_map[i].items()}
                                    for i in range(hierarchy_levels)]
    
    @property
    def offset(self) -> List[int]:
        return self._offset
    @offset.setter
    def offset(self, value: List[int]) -> None:
        self._offset = value
        # Update predIndex_to_targetId, targetId_to_predIndex and classes if already set
        # to respect the new offset
        if getattr(self, "targetId_to_predIndex", None) is not None:
            self.targetId_to_predIndex = [{targetID:predIndex+self._offset[i] for predIndex, targetID in enumerate(sorted(list(level.keys())))}
                                            for i, level in enumerate(self.id_to_name)]
        if getattr(self, "predIndex_to_targetId", None) is not None:
            self.predIndex_to_targetId = [{predIndex+self._offset[i]:targetID for predIndex, targetID in enumerate(sorted(list(level.keys())))}
                                            for i, level in enumerate(self.id_to_name)]
        if getattr(self, "classes", None) is not None:
            self.classes = [[self.id_to_name[level][self.predIndex_to_targetId[level][class_idx+self._offset[level]]] 
                                for class_idx in range(self.num_classes[level])] 
                            for level in range(self.hierarchy_levels)]
        if getattr(self, "pred_coarse_fine_map", None) is not None:
            self.pred_coarse_fine_map = [{self.targetId_to_predIndex[i][k]:self.targetId_to_predIndex[-1][v] for k,v in self.coarse_fine_map[i].items()} 
                                         for i in range(self.hierarchy_levels)]
        if getattr(self, "pred_fine_coarse_map", None) is not None:
            self.pred_fine_coarse_map = [{self.targetId_to_predIndex[-1][k]:self.targetId_to_predIndex[i][v] for k,v in self.fine_coarse_map[i].items()}
                                            for i in range(self.hierarchy_levels)]
    
    # Use properties to hide the global variables (needed for offset update)     
    @property
    def predIndex_to_targetId(self) -> List[Dict[int, int]]:
        return self._predIndex_to_targetId
    @predIndex_to_targetId.setter
    def predIndex_to_targetId(self, value: List[Dict[int, int]]) -> None:
        self._predIndex_to_targetId = value
    
    @property
    def targetId_to_predIndex(self) -> List[Dict[int, int]]:
        return self._targetId_to_predIndex
    @targetId_to_predIndex.setter
    def targetId_to_predIndex(self, value: List[Dict[int, int]]) -> None:
        self._targetId_to_predIndex = value
        
    @property
    def pred_coarse_fine_map(self) -> List[Dict[int, int]]:
        return self._pred_coarse_fine_map
    @pred_coarse_fine_map.setter
    def pred_coarse_fine_map(self, value: List[Dict[int, int]]) -> None:
        self._pred_coarse_fine_map = value
        
    @property
    def pred_fine_coarse_map(self) -> List[Dict[int, int]]:
        return self._pred_fine_coarse_map
    @pred_fine_coarse_map.setter
    def pred_fine_coarse_map(self, value: List[Dict[int, int]]) -> None:
        self._pred_fine_coarse_map = value
        
    @property
    def fine_coarse_name_map(self) -> List[Dict[int, int]]:
        return self._fine_coarse_name_map
    @fine_coarse_name_map.setter
    def fine_coarse_name_map(self, value: List[Dict[int, int]]) -> None:
        self._fine_coarse_name_map = value
        
    @property
    def classes(self) -> List[List[str]]:
        return self._classes
    @classes.setter
    def classes(self, value: List[List[str]]) -> None:
        self._classes = value
        
    @property
    def hierarchy_level_names(self) -> List[str]:
        return self._hierarchy_level_names
    @hierarchy_level_names.setter
    def hierarchy_level_names(self, value: List[str]) -> None:
        if len(value) > self.hierarchy_levels:
            print(f"Number of level names ({len(value)}) exceeds hierarchy levels of descriptor ({self.hierarchy_levels}). "
                  f"Only the last hierarchy levels ({','.join(value[-self.hierarchy_levels:])}) will be used.")
        elif len(value) < self.hierarchy_levels:
            print(f"Number of level names ({len(value)}) is less than hierarchy levels of descriptor ({self.hierarchy_levels}). "
                  "Not updating names and using current names.")
            return
        # At most as many names as levels (note coarse to fine structure of hierarchy)
        self._hierarchy_level_names = value[-self.hierarchy_levels:]
        self.default_names = False
        
    def update_classes(self) -> None:
        self.classes = [[self.id_to_name[level][self.predIndex_to_targetId[level][class_idx+self._offset[level]]] 
                                for class_idx in range(self.num_classes[level])] 
                            for level in range(self.hierarchy_levels)]
        
    def update_coarse_fine_pred_map(self) -> None:
        self.pred_coarse_fine_map = [{self.targetId_to_predIndex[i][k]:self.targetId_to_predIndex[-1][v] for k,v in self.coarse_fine_map[i].items()} 
                                         for i in range(self.hierarchy_levels)]
        self.pred_fine_coarse_map = [{self.targetId_to_predIndex[-1][k]:self.targetId_to_predIndex[i][v] for k,v in self.fine_coarse_map[i].items()}
                                            for i in range(self.hierarchy_levels)]
        # Make sure to call update_classes before this if that changed as well.
        self.fine_coarse_name_map = [{model:self.classes[i][make] 
                                        for model, make in self.pred_fine_coarse_map[i].items()}
                                    for i in range(self.hierarchy_levels)]
        
    def parse_data_file(self, 
                        file_name: str, 
                        main_sep: str = '=', 
                        sub_sep: str = "#") -> List[Tuple[List[int], List[str]]]:
        """Parse file to data list with custom Dataset Descriptor file Format
        Expected format:
        [ClassIndices]`main_sep`[ClassNames]

        Within [ClassIndices] and [ClassNames] multiple fields can be set by `sub_sep` separator. 

        Args:
            file_name (str): The path of data file
            main_sep (str): Main separator to split fields by. Defaults to '='
            sub_sep (str): Separator to split within fields. Defaults to '#'
            
        Returns:
            List(Tuple[List[int], List[str]]): List of (class_indices, class_names) tuples
        """
        with open(file_name, "r", encoding='utf8') as f:
            data_list = []
            for line in f.read().splitlines():
                split_line = line.split(sep=main_sep)
                indices = [int(i) for i in split_line[0].split(sep=sub_sep)]
                names = [i for i in split_line[1].split(sep=sub_sep)]
                data_list.append((indices, names))
        return data_list
    
    def __eq__(self, other) : 
        return self.__dict__ == other.__dict__
    
    @property
    def targetIDs(self) -> List[Set[int]]:
        return [set(level.keys()) for level in self.id_to_name]

class CustomDataset(datasets.VisionDataset):
    """A generic Dataset class for image classification
    Adapted from tllib.vision.datasets.imagelist.ImageList
    
    Probably should not be used directly but rather subclassed as this class is not tested standalone.

    Args:
        root (str): Root directory of dataset
        annfile_path (str): File to read the image list from.
        descriptor (Optional[DatasetDescriptor], optional): Descriptor for the dataset. 
            If given has the highest priority. Defaults to None.
        descriptor_file (Optional[str], optional): File to construct the descriptor from. 
            Higher priority than `classes` but will not overwrite `Descriptor`. Defaults to None.
        hierarchy_levels (Optional[int], optional): Number of hierarchy levels in the descriptor. 
            Only relevant when using `descriptor_file`. 
             If not given, the number is inferred based on the descriptor file. Defaults to None.
        classes (list[str]): The names of all the classes. Only used if `descriptor` and `descriptor_file` is None. 
            Defaults to None.
        transform (callable, optional): A function/transform that takes in an PIL image 
            and returns a transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `annfile_path`, each line has 2 values in the following format.
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your annfile_path has different formats, please over-ride :meth:`~CustomDataset.parse_data_file`.
    """
    multi_label = False
    def __init__(self, 
                 root: str, 
                 annfile_path: str,
                 descriptor: Optional[DatasetDescriptor] = None,
                 descriptor_file: Optional[str] = None,
                 hierarchy_levels: Optional[int] = None,
                 label_index: bool = False,
                 classes: Optional[List[str]] = None, 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self.parse_data_file(annfile_path)
        self.targets = [s[1] for s in self.samples]
        self.label_index = defaultdict(list)
        if label_index:
            self.label_index = self.build_label_index()
        
        self.loader = default_loader
        self.annfile_path = annfile_path
        if descriptor is not None:
            if getattr(self, 'hierarchy_level_names', False) and descriptor.default_names:
                # Property setter will set default_names to False
                descriptor.hierarchy_level_names = self.hierarchy_level_names
            self.descriptor = descriptor
            self.classes = descriptor.classes
        elif descriptor_file is not None:
            self.descriptor = DatasetDescriptor(descriptor_file, hierarchy_levels, 
                                                level_names=getattr(self, 'hierarchy_level_names', None))
            self.classes = self.descriptor.classes
        elif classes is not None:
            self.classes = classes
        else:
            raise ValueError("No descriptor, descriptor_file or classes provided.")
        
    @property 
    def eval_classes(self) -> List[int]:
        return self.targets

    def __getitem__(self, index: int) -> Tuple[Any, int | Sequence[int]]:
        """
        Args:
            index (int): Index
            return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.samples)
    
    def build_label_index(self) -> defaultdict:
        index = defaultdict(list)
        if hasattr(self, 'main_class_idx'):
            for i, target in enumerate(self.targets):
                index[target[self.main_class_idx]].append(i)
        else:
            try:
                for i, target in enumerate(self.targets):
                    index[target].append(i)
            except:
                # Do nothing and return empty index
                index = defaultdict(list)
        return index

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Args:
            file_name (str): The path of data file
            return (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                split_line = line.split()
                target = split_line[-1]
                path = ' '.join(split_line[:-1])
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def num_classes(self) -> List[int]:
        """Number of classes"""
        return self.descriptor.num_classes
    
    @classmethod
    def dataset_kwargs(cls, **kwargs) -> dict:
        """
        Creates dictionary that filteres relevant kwargs for dataset
        """
        dataset_kwargs = {}
        # Using the inspect module to get the arguments of the constructor
        # This way we can filter out the relevant arguments without hardcoding them
        # and thus no need to update this method when the constructor changes
        # or in subclasses
        arguments = inspect.getfullargspec(cls.__init__).args
        # Only add if given otherwise use default from constructor
        for arg in arguments:
            if arg in kwargs:
                dataset_kwargs[arg] = kwargs[arg]
        return dataset_kwargs