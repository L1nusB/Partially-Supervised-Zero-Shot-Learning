"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
from contextlib import suppress
from functools import partial
import os.path as osp
import os
import psutil
from typing import Optional, Sequence, Tuple, overload
import argparse
import ast
import warnings
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as col

import torch
import torch.nn as nn  
import torch.nn.functional as F
from torch.utils.data import DataLoader

from timm.data import Mixup, FastCollateMixup
from timm.scheduler.scheduler import Scheduler
from timm.utils import NativeScaler
from timm.utils import dispatch_clip_grad

class NativeScalerMultiple(NativeScaler):
    def __call__(
            self,
            loss,
            optimizers,
            clip_grad=None,
            clip_mode='norm',
            parameters=None,
            create_graph=False,
            need_update=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if need_update:
            for optimizer in optimizers:
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
                self._scaler.step(optimizer)
            self._scaler.update()

# Do not show source line for warnings
# warnings.formatwarning = lambda message, *args, **kwargs: f'{message}\n'

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            # replace hyphens with underscores for argparse namespace
            # for uniform representations in code
            key = key.replace('-', '_')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)
        
def setup_device(args: Optional[argparse.Namespace]=None, 
                 device_key: str = "device", 
                 device: Optional[str]=None) -> torch.device:
    if device is None:
        assert args is not None, "Either args or device must be provided."
        device = getattr(args, device_key, "cpu")
    
    if torch.cuda.is_available() == False:
        warnings.warn("CUDA is not available, using CPU.")
        device = torch.device('cpu')
    elif device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def send_to_device(tensor: torch.Tensor, device) -> torch.Tensor:
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)

def calculate_learning_rate(args:argparse.Namespace) -> float:
    # If lr is set directly do not touch
    if args.lr:
        print(f'Learning rate from argument is taken: {args.lr}')
        return args.lr

    accum_batch_size = args.batch_size * args.grad_accum_steps
    batch_ratio = accum_batch_size / args.lr_base_size
    if not args.lr_base_scale:
        used_optimizer = args.opt.lower()
        args.lr_base_scale = 'sqrt' if any([o in used_optimizer for o in ('ada', 'lamb')]) else 'linear'
    if args.lr_base_scale == 'sqrt':
        batch_ratio = batch_ratio ** 0.5
    lr = args.lr_base * batch_ratio
    print(f'Learning rate ({lr}) calculated from base learning rate ({args.lr_base}), '
          f'base learning rate batch size ({args.lr_base_size}), batch ratio ({batch_ratio}) '
          f'and effective global batch size ({accum_batch_size}) with {args.lr_base_scale} scaling.')
    return lr

def preview_lr_schedule(scheduler: Scheduler, num_steps: int = None):
    if num_steps is None:
        num_steps = scheduler.t_initial
    for i in range(num_steps):
        print(f"{i}: {scheduler._get_lr(i)}")

@overload
def setup_mixup_cutmix(args: argparse.Namespace,
                       num_classes: int) -> Tuple[FastCollateMixup, Mixup]:...
@overload
def setup_mixup_cutmix(args: argparse.Namespace,
                       num_classes: Sequence[int]) -> Tuple[Sequence[FastCollateMixup], Sequence[Mixup]]:...
def setup_mixup_cutmix(args: argparse.Namespace,
                       num_classes: int | Sequence[int]
                       ) -> Tuple[FastCollateMixup, Mixup] | Tuple[Sequence[FastCollateMixup], Sequence[Mixup]]:
    # setup mixup / cutmix
    single_return = isinstance(num_classes, int)
    if single_return:
        num_classes = [num_classes]
    collates = []
    mixups = []
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        # Due to the way the datasets are actually zero-indexed in ascending order this should work anyways
        warnings.warn("Mixup/Cutmix not tested/validated with internal mapping functions.")
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
        )
        for nc in num_classes:
            mixup_args['num_classes'] = nc
            if not args.no_prefetch:
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)
            collates.append(collate_fn)
            mixups.append(mixup_fn)
    else:
        return None, None
        
    if single_return:
        return collates[0], mixups[0]
    else:
        return collates, mixups

def setup_amp(args: argparse.Namespace,
              device: torch.device) -> Tuple[suppress | torch.autocast, None | NativeScalerMultiple]:
    # Setup amp for mixed precision training
    amp_autocast = suppress
    loss_scaler = None
    if args.amp:
        print(f"Using native torch AMP for mixed precision training with dtype {args.amp_dtype}.")
        if args.amp_dtype == 'float16':
            amp_dytpe = torch.float16
        elif args.amp_dtype == 'bfloat16':
            amp_dytpe = torch.bfloat16
        else:
            warnings.warn(f"AMP dtype {args.amp_dtype} not supported. Using bfloat16 instead.")
            amp_dytpe = torch.bfloat16
        try:    
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dytpe)
        except (AttributeError, TypeError):
            # fallback to CUDA only AMP for PyTorch < 1.10
            assert device.type == 'cuda'
            amp_autocast = torch.cuda.amp.autocast
        
        if device.type == 'cuda' and amp_dytpe == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScalerMultiple()
    else:
        print("AMP not enabled. Training in float32.")
    return amp_autocast, loss_scaler

def visualize_image(img):
    import matplotlib.pyplot as plt
    if isinstance(img, Sequence):
        img = img[0]
    for i in range(len(img)):
        plt.imshow(img.cpu()[i].permute(1,2,0))

def get_images_dataset(data_iter, 
                       num_batches: Optional[int] = None, 
                       save_dir: Optional[str] = None):
    
    import logging
    if num_batches is None:
        num_batches = len(data_iter)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for i in range(len(data_iter)):
        dat, _ = next(data_iter)
        # Terminate if num_batches is reached
        if i > num_batches:
            break
        if save_dir:
            save_filepath = osp.join(save_dir, f'batch_{i}.png')
        logger = logging.getLogger()
        old_level = logger.level
        logger.setLevel(100)
        display_images(dat, save_filepath)
        logger.setLevel(old_level)

def display_images(images: Sequence[torch.Tensor], 
                   save_filepath: Optional[str] = None):
    import matplotlib.pyplot as plt
    # Turn interactive plotting off
    plt.ioff()
    num_images = len(images)
    num_cols = int(num_images ** 0.5)
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    for i, img in enumerate(images):
        row = i // num_cols
        col = i % num_cols

        if isinstance(img, torch.Tensor):
            img = img.cpu()
            
        axes[row, col].imshow(img.permute(1, 2, 0))
        axes[row, col].axis('off')

    if save_filepath:
        plt.savefig(save_filepath)
    else:
        plt.show()

def is_other_sh_scripts_running():
    current_pid = os.getpid()
    current_script = os.path.basename(__file__)
    
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Process info
            pid = process.info['pid']
            cmdline = process.info['cmdline']
            
            # Check if the process is a shell script and not the current script
            if pid != current_pid and cmdline and (cmdline[0] == 'sh' or cmdline[-1].endswith('.sh')):
                script_name = os.path.basename(cmdline[-1])
                if script_name != current_script:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return False

def tSNE_A_distance(source_loader: DataLoader, 
                    target_loader: DataLoader,
                    model: nn.Module, device: torch.device,
                    eval_classes: Sequence[int], out_dir: str) -> None:
    sourceFeatures_base = []
    sourceFeatures_novel = []
    targetFeatures_base = []
    targetFeatures_novel = []
    model.eval()
    
    with torch.no_grad():
        evals = torch.tensor(eval_classes, device=device)
        for i, (data, labels) in enumerate(target_loader):
            k = torch.isin(labels, evals)
            if any(k) == False:
                # No Novel classes in this batch
                targetFeatures_base.append(model(data).cpu())
            else:
                datBase = data[~k]
                datNovel = data[k]
                targetFeatures_base.append(model(datBase).cpu())
                targetFeatures_novel.append(model(datNovel).cpu())
        for i, (data, labels) in enumerate(source_loader):
            k = torch.isin(labels, evals)
            if any(k) == False:
                # No Novel classes in this batch
                sourceFeatures_base.append(model(data).cpu())
            else:
                datBase = data[~k]
                datNovel = data[k]
                sourceFeatures_base.append(model(datBase).cpu())
                sourceFeatures_novel.append(model(datNovel).cpu())
                
        targetFeatures_base = torch.cat(targetFeatures_base, dim=0)
        targetFeatures_novel = torch.cat(targetFeatures_novel, dim=0)
        targetFeatures = torch.cat([targetFeatures_base, targetFeatures_novel], dim=0)
        sourceFeatures_base = torch.cat(sourceFeatures_base, dim=0)
        sourceFeatures_novel = torch.cat(sourceFeatures_novel, dim=0)
        sourceFeatures = torch.cat([sourceFeatures_base, sourceFeatures_novel], dim=0)

        tSNE_visualize_binary(sourceFeatures, targetFeatures, os.path.join(out_dir, 'tSNEBin.pdf'))
        tSNE_visualize_binary(sourceFeatures, targetFeatures, os.path.join(out_dir, 'tSNEBinR.pdf'), reduce_dim=50)
        # tSNE_visualize_binary(sourceFeatures, targetFeatures, os.path.join(out_dir, 'tSNEBinRep.pdf'), reduce_dim=50, reduce_separate=True)
        tSNE_visualize_fourway(sourceFeatures_base, sourceFeatures_novel,
                               targetFeatures_base, targetFeatures_novel,
                               os.path.join(out_dir, 'tSNEFour.pdf'),)
        tSNE_visualize_fourway(sourceFeatures_base, sourceFeatures_novel,
                               targetFeatures_base, targetFeatures_novel,
                               os.path.join(out_dir, 'tSNEFourR.pdf'),
                               reduce_dim=50)
        # tSNE_visualize_fourway(sourceFeatures_base, sourceFeatures_novel,
        #                        targetFeatures_base, targetFeatures_novel,
        #                        os.path.join(out_dir, 'tSNEFourRSep.pdf'),
        #                        reduce_dim=50, reduce_separate=True)
    A_distance_base = calculateADist(sourceFeatures_base, targetFeatures_base, device, progress=False)
    A_distance_novel = calculateADist(sourceFeatures_novel, targetFeatures_novel, device, progress=False)
    A_distance_total = calculateADist(sourceFeatures, targetFeatures, device, progress=False)
    print(f'A-Distance (base): {A_distance_base}, A-Distance (novel): {A_distance_novel}, A-Distance (total): {A_distance_total}')
    
    with open(os.path.join(out_dir, 'A-Distance.txt'), 'w') as f:
        f.write(f'A-Distance (base): {A_distance_base} \n')
        f.write(f'A-Distance (novel): {A_distance_novel} \n')
        f.write(f'A-Distance (total): {A_distance_total} \n')
    

def tSNE_visualize_binary(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b',
              reduce_dim: Optional[int] = None, reduce_separate: bool = False):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    if reduce_dim is not None:
        if reduce_separate:
            source_feature = PCA(n_components=reduce_dim).fit_transform(source_feature)
            target_feature = PCA(n_components=reduce_dim).fit_transform(target_feature)
            features = np.concatenate([source_feature, target_feature], axis=0)
        else:
            features = np.concatenate([source_feature, target_feature], axis=0)
            features = PCA(n_components=reduce_dim).fit_transform(features)
    else:
        features = np.concatenate([source_feature, target_feature], axis=0)
    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)
    
def tSNE_visualize_fourway(source_feature_base: torch.Tensor,
                           source_feature_novel: torch.Tensor, 
                           target_feature_base: torch.Tensor,
                           target_feature_novel: torch.Tensor,
                           filename: str, 
                           source_base_color='r', 
                           source_novel_color='g', 
                           target_base_color='b',
                           target_novel_color='m',
                           base_shape = 'o',
                           novel_shape = '^',
                           reduce_dim: Optional[int] = None,
                           reduce_separate: bool = False):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature_base = source_feature_base.numpy()
    target_feature_base = target_feature_base.numpy()
    source_feature_novel = source_feature_novel.numpy()
    target_feature_novel = target_feature_novel.numpy()
    if reduce_dim is not None:
        if reduce_separate:
            source_feature_base = PCA(n_components=reduce_dim).fit_transform(source_feature_base)
            source_feature_novel = PCA(n_components=reduce_dim).fit_transform(source_feature_novel)
            target_feature_base = PCA(n_components=reduce_dim).fit_transform(target_feature_base)
            target_feature_novel = PCA(n_components=reduce_dim).fit_transform(target_feature_novel)
            features_base = np.concatenate([source_feature_base, target_feature_base], axis=0)
            features_novel = np.concatenate([source_feature_novel, target_feature_novel], axis=0)
        else:
            features_base = np.concatenate([source_feature_base, target_feature_base], axis=0)
            features_novel = np.concatenate([source_feature_novel, target_feature_novel], axis=0)
            features_base = PCA(n_components=reduce_dim).fit_transform(features_base)
            features_novel = PCA(n_components=reduce_dim).fit_transform(features_novel)
    else:
        features_base = np.concatenate([source_feature_base, target_feature_base], axis=0)
        features_novel = np.concatenate([source_feature_novel, target_feature_novel], axis=0)
    
    # map features to 2-d using TSNE
    X_tsne_base = TSNE(n_components=2, random_state=33).fit_transform(features_base)
    X_tsne_novel = TSNE(n_components=2, random_state=33).fit_transform(features_novel)

    # domain labels, 1 represents source while 0 represents target
    domains_base = np.concatenate((np.ones(len(source_feature_base)), np.zeros(len(target_feature_base))))
    domains_novel = np.concatenate((np.ones(len(source_feature_novel)), np.zeros(len(target_feature_novel))))

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax: plt.Axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.scatter(X_tsne_base[:, 0], X_tsne_base[:, 1], marker=base_shape, c=domains_base, cmap=col.ListedColormap([target_base_color, source_base_color]), s=20)
    ax.scatter(X_tsne_novel[:, 0], X_tsne_novel[:, 1], marker=novel_shape, c=domains_novel, cmap=col.ListedColormap([target_novel_color, source_novel_color]), s=20)
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)
    
class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x

def calculateADist(source_feature: torch.Tensor, target_feature: torch.Tensor,
              device, progress=True, training_epochs=10) -> float:
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.

    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier

    Returns:
        :math:`\mathcal{A}`-distance
    """
    from torch.utils.data import TensorDataset
    from torch.optim import SGD
    from PSZS.Metrics import binary_accuracy
    
    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))
    feature = torch.cat([source_feature, target_feature], dim=0)
    label = torch.cat([source_label, target_label], dim=0)

    dataset = TensorDataset(feature, label)
    length = len(dataset)
    train_size = int(0.8 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    anet = ANet(feature.shape[1]).to(device)
    optimizer = SGD(anet.parameters(), lr=0.01)
    a_distance = 2.0
    for epoch in range(training_epochs):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        anet.eval()
        accAccum = 0
        normalizer = 0
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                acc = binary_accuracy(y, label)
                accAccum += acc * x.shape[0]
                normalizer += x.shape[0]
        accAvg = accAccum / normalizer
        error = 1 - accAvg / 100
        a_distance : torch.Tensor = 2 * (1 - 2 * error)
        if progress:
            print("epoch {} accuracy: {} A-dist: {}".format(epoch, accAvg, a_distance))

    return a_distance.item()