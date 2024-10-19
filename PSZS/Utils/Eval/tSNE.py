from typing import Optional, Sequence
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
from itertools import cycle, islice

from PSZS.Utils.utils import filter_kwargs

@torch.no_grad()
def tSNE_visualize(*features: torch.Tensor|np.ndarray, 
                   fName: str = 'tSNE.pdf',
                   labels: Optional[torch.Tensor|np.ndarray]=None, 
                   reduce_dim: Optional[int] = 50,
                   markers: Optional[str|Sequence[str]] = 'o',
                   colors: Optional[str|Sequence[str]] = None,
                   despine: bool = True,
                   save: bool = True,
                   cluster: Optional[int] = None,
                   **kwargs) -> None:
    """
    Visualizes high-dimensional data using t-SNE (t-distributed Stochastic Neighbor Embedding).
    
    Args:
        features (torch.Tensor|np.ndarray): High-dimensional data to visualize.
        fName (str): Filename to save the plot.
        labels (torch.Tensor|np.ndarray, optional): Labels for each feature for coloring. Defaults to None.
        reduce_dim (int, optional): Number of dimensions to reduce to using PCA before t-SNE. Defaults to 50.
        markers (str|Sequence[str], optional): Markers for each feature type. Defaults to 'o'.
        colors (str|Sequence[str], optional): Colors for each feature type. Defaults to None.
        despine (bool): Whether to remove spines from the plot. Defaults to True.
        save (bool): Whether to save the plot. Defaults to True.
        **kwargs: Additional keyword arguments for customization.
    """
    # Create cluster assignment and set labels
    clusters = np.concatenate([np.full(len(feat), i) for i, feat in enumerate(features)])
    if labels is None:
        labels = [f'Cluster {i}' for i in clusters]
    else:
        assert len(labels) == len(features), f"Labels must match the number of features. Got {len(labels)} != {len(features)}."
    
    if markers is not None and len(markers) != len(features):
        warnings.warn(f"Markers must match the number of features. Got {len(markers)} != {len(features)}. Repeating markers.")
        markers = list(islice(cycle(markers), len(features)))
    if colors is not None and len(colors) != len(features):
        warnings.warn(f"Colors must match the number of features. Got {len(colors)} != {len(features)}. Repeating colors.")
        colors = list(islice(cycle(colors), len(features)))
        
    if fName[-4:] != '.png' and fName[-4:] != '.jpg' and fName[-4:] != '.pdf':
        fName += '.pdf'
    
    # Combine features into a single numpy array
    features = np.concatenate([feat.cpu().numpy() for feat in features], axis=0)
    
    if reduce_dim is not None:
        # Reduce dimensionality using PCA before t-SNE
        pca = PCA(n_components=reduce_dim)
        features = pca.fit_transform(features)

    # Initialize t-SNE with random state for reproducability and fit features
    tSNE_kwargs = filter_kwargs(TSNE, kwargs)
    rand_state = tSNE_kwargs.pop('random_state', 42)
    n_components = tSNE_kwargs.pop('n_components', 2)
    tsne = TSNE(n_components=n_components, random_state=rand_state, **tSNE_kwargs).fit_transform(features)
    
    fig_kwargs = filter_kwargs(plt.subplots, kwargs)
    fig_size = fig_kwargs.pop('figsize', (10,10))
    # Create a scatter plot
    if cluster is not None:
        fig, axs = plt.subplots(1, 3, figsize=(3*fig_size[0], fig_size[1]), **fig_kwargs)
        ax = axs[0]
        cls_ax = axs[1]
        cls_ax2 = axs[2]
        cls_ax: plt.Axes
        cls_ax2: plt.Axes
    else:
        fig, ax = plt.subplots(figsize=fig_size, **fig_kwargs)
    ax: plt.Axes
    if despine:
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
    scatter_kwargs = filter_kwargs(plt.scatter, kwargs)
    s = scatter_kwargs.pop('s', 50)
    edgecolors = scatter_kwargs.pop('edgecolors', 'white')
    linewidth = scatter_kwargs.pop('linewidth', 0.5)
    # Create scatter plots for each type
    for i, label in enumerate(labels):
        mask = clusters == i
        ax.scatter(tsne[mask, 0], tsne[mask, 1], 
                    c=[colors[i]], marker=markers[i], s=s, 
                    label=label, edgecolors=edgecolors, linewidth=linewidth,
                    **scatter_kwargs)

    # Add legend
    ax.legend(title='Feature Types', loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title(kwargs.get('title', 't-SNE Visualization'), 
              fontsize=kwargs.get('fontsize', 16),
              pad=kwargs.get('pad', 20))
    if despine:
        ax.set_xticks([])
        ax.set_yticks([])
    if cluster is not None:
        # Perform clustering using KMeans
        kmeans = KMeans(n_clusters=cluster, random_state=rand_state).fit(tsne)
        scat = cls_ax.scatter(tsne[:, 0], tsne[:, 1], c=kmeans.labels_, cmap='viridis', 
                       s=s, edgecolors=edgecolors, linewidth=linewidth)
        cls_ax.set_title(f'KMeans Clustering (k={cluster})', fontsize=kwargs.get('fontsize', 16), pad=kwargs.get('pad', 20))
        if despine:
            cls_ax.set_xticks([])
            cls_ax.set_yticks([])
        fig.colorbar(scat, ax=cls_ax)
        
        kmeans = KMeans(n_clusters=cluster, random_state=rand_state).fit_predict(features)
        scat2 = cls_ax2.scatter(tsne[:, 0], tsne[:, 1], c=kmeans, cmap='viridis', 
                        s=s, edgecolors=edgecolors, linewidth=linewidth)
        cls_ax2.set_title(f'KMeans Clustering 2 (k={cluster})', fontsize=kwargs.get('fontsize', 16), pad=kwargs.get('pad', 20))
        fig.colorbar(scat2, ax=cls_ax2)
        if despine:
            cls_ax.set_xticks([])
            cls_ax.set_yticks([])
    plt.tight_layout()
    if save:
        plt.savefig(fName)
    else:
        plt.show()
    plt.close()
    
@torch.no_grad()
def PCA_visualize(*features: torch.Tensor|np.ndarray, 
                   fName: str = 'PCA.pdf',
                   labels: Optional[torch.Tensor|np.ndarray]=None, 
                   reduce_dim: Optional[int] = None,
                   markers: Optional[str|Sequence[str]] = 'o',
                   colors: Optional[str|Sequence[str]] = None,
                   despine: bool = True,
                   save: bool = True,
                   **kwargs) -> None:
    """
    Visualizes high-dimensional data using PCA (Principal Component Analysis).
    
    Args:
        features (torch.Tensor|np.ndarray): High-dimensional data to visualize.
        fName (str): Filename to save the plot. Defaults to 'PCA.pdf'.
        labels (torch.Tensor|np.ndarray, optional): Labels for each feature for coloring. Defaults to None.
        reduce_dim (int, optional): Number of dimensions to reduce to using PCA before reducing down to 2 for visualization.
        markers (str|Sequence[str], optional): Markers for each feature type. Defaults to 'o'.
        colors (str|Sequence[str], optional): Colors for each feature type. Defaults to None.
        despine (bool): Whether to remove spines from the plot. Defaults to True.
        save (bool): Whether to save the plot. Defaults to True.
        **kwargs: Additional keyword arguments for customization.
    """
    # Create cluster assignment and set labels
    clusters = np.concatenate([np.full(len(feat), i) for i, feat in enumerate(features)])
    if labels is None:
        labels = [f'Cluster {i}' for i in clusters]
    else:
        assert len(labels) == len(features), f"Labels must match the number of features. Got {len(labels)} != {len(features)}."
    
    if markers is not None and len(markers) != len(features):
        warnings.warn(f"Markers must match the number of features. Got {len(markers)} != {len(features)}. Repeating markers.")
        markers = list(islice(cycle(markers), len(features)))
    if colors is not None and len(colors) != len(features):
        warnings.warn(f"Colors must match the number of features. Got {len(colors)} != {len(features)}. Repeating colors.")
        colors = list(islice(cycle(colors), len(features)))
        
    if fName[-4:] != '.png' and fName[-4:] != '.jpg' and fName[-4:] != '.pdf':
        fName += '.pdf'
    
    # Combine features into a single numpy array
    features = np.concatenate([feat.cpu().numpy() for feat in features], axis=0)
    
    if reduce_dim is not None:
        # Reduce dimensionality using PCA before t-SNE
        pca = PCA(n_components=reduce_dim)
        features = pca.fit_transform(features)
    
    pca_kwargs = filter_kwargs(PCA, kwargs)
    rand_state = pca_kwargs.pop('random_state', 42)
    pca = PCA(n_components=2, random_state=rand_state,).fit_transform(features)
    
    
    fig_kwargs = filter_kwargs(plt.subplots, kwargs)
    fig_size = fig_kwargs.pop('figsize', (10,10))
    # Create a scatter plot
    fig, ax = plt.subplots(figsize=fig_size, **fig_kwargs)
    ax: plt.Axes
    if despine:
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
    scatter_kwargs = filter_kwargs(plt.scatter, kwargs)
    s = scatter_kwargs.pop('s', 50)
    edgecolors = scatter_kwargs.pop('edgecolors', 'white')
    linewidth = scatter_kwargs.pop('linewidth', 0.5)
    # Create scatter plots for each type
    for i, label in enumerate(labels):
        mask = clusters == i
        plt.scatter(pca[mask, 0], pca[mask, 1], 
                    c=[colors[i]], marker=markers[i], s=s, 
                    label=label, edgecolors=edgecolors, linewidth=linewidth,
                    **scatter_kwargs)

    # Add legend
    plt.legend(title='Feature Types', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(kwargs.get('title', 'PCA Visualization'), 
              fontsize=kwargs.get('fontsize', 16),
              pad=kwargs.get('pad', 20))
    if despine:
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    if save:
        plt.savefig(fName)
    else:
        plt.show()
    plt.close()