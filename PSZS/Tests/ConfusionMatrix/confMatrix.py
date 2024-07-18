import argparse
import numpy as np

from PSZS.Metrics import ConfusionMatrix
from PSZS.datasets import build_remapped_descriptors
from PSZS.Tests.utils import setup_seed, HIERARCHY_LEVELS

def test_confmatrix(args):
    setup_seed(args)
    total_descriptor, _, _ = build_remapped_descriptors(fileRoot=args.root, ds_split=args.ds_split)
    if isinstance(args.hierarchy_level, str):
        args.hierarchy_level = HIERARCHY_LEVELS.get(args.hierarchy_level.lower(), None)
        if args.hierarchy_level is None:
            raise ValueError(f"Unknown hierarchy level {args.hierarchy_level}. "
                             f"Available values are: {','.join(HIERARCHY_LEVELS.keys())}.")
    num_classes = total_descriptor.num_classes[args.hierarchy_level]
    confMat = ConfusionMatrix(num_classes=num_classes, class_names=total_descriptor.classes[args.hierarchy_level])
    confMat.mat = np.random.randint(3, size=(num_classes, num_classes))
    # confMat.create_report(fileName='confusionMatrixTest', 
    #                     start_class=-5, show_class_names=args.display_names)
    test_map = {model:total_descriptor.classes[0][make] for model, make in total_descriptor.pred_fine_coarse_map[0].items() }
    confMat.save_reduced_conf(last_relevant=-56, path='reducedConfusionMatrix.xlsx', 
                              class_names=total_descriptor.classes[args.hierarchy_level],
                              secondary_info=total_descriptor.fine_coarse_name_map[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test for different eval mode mapping modes')
    parser.add_argument('root', metavar='DIR', help='root path of dataset')
    parser.add_argument('ds_split', type=str, default=None,
                       help='Which split of the dataset should be used. Gets appended to annfile dir. Default None.')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--hierarchy-level', default=-1, 
                        help='Hierarchy level that the generated test objects should represent. Default -1.'
                        'See HIERARCHY_LEVELS for available values. Can also be given as string.')
    parser.add_argument('--display-names', action='store_true',
                        help='Display class names in confusion matrix. ')
    args = parser.parse_args()
    test_confmatrix(args)