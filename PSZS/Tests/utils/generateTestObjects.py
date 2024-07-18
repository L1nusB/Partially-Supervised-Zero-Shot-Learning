from typing import Sequence, overload, Tuple

import torch

@overload
def generate_random_tensor(size: Sequence[int], 
                           min: Sequence[int], 
                           max: Sequence[int],
                           separate_tensors: bool = True) -> Tuple[torch.Tensor,...]:...
@overload
def generate_random_tensor(size: int| Sequence[int] = 10, 
                           min: int| Sequence[int] = 0, 
                           max: int | Sequence[int] = 281,
                           separate_tensors: bool = False) -> torch.Tensor:...

def generate_random_tensor(size: int| Sequence[int] = 10, 
                           min: int| Sequence[int] = 0, 
                           max: int | Sequence[int] = 281,
                           separate_tensors: bool = False) -> torch.Tensor|Tuple[torch.Tensor,...]:
    """Generates a tensor of random integers. 
    Multiple sizes, mins and maxes can be provided to generate a 2d-tensor.
    While not all values have to be sequences (e.g. size can be an int), 
    the number of elements in each sequence must be the same.
    If `separate_tensors` is True, a tuple of 1d tensors is returned.

    Args:
        size (int | Sequence[int], optional): Number of elements for each axis. Defaults to 10.
        min (int | Sequence[int], optional): Minimal value for elements. Defaults to 0.
        max (int | Sequence[int], optional): Maximal value for elements. Defaults to 281.
        separate_tensors (bool, optional): Whether to return a tuple of tensors. Defaults to False.

    Returns:
        torch.Tensor | Tuple[torch.Tensor,...]: Random tensor of integers or tuple of random 1d tensors of integers.
    """
    count = None
    if isinstance(size, int) == False:
        count = len(size)
    if isinstance(min, int) == False:
        if count is not None:
            assert count == len(min), f"Size and min must have the same length. But got {count} and {len(min)}."
        else:
            count = len(min)
    if isinstance(max, int) == False:
        if count is not None:
            assert count == len(max), f"Size and max must have the same length. But got {count} and {len(max)}."
        else:
            count = len(max)
    
    if count is None:
        count = 1
    # Expand single values to lists
    if isinstance(min, int):
        min = [min] * count
    if isinstance(max, int):
        max = [max] * count
    if isinstance(size, int):
        size = [size] * count
    elements = [torch.randint(low=min[i], high=max[i], size=(size[i],), dtype=torch.int32) for i in range(count)]
    if separate_tensors:
        return tuple(elements)
    else:
        return torch.stack(elements).squeeze()