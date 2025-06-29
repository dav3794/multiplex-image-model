import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from math import ceil, pi, cos
from typing import Dict, List, Literal
from nltk.tree import Tree
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_val=-15.0, max_val=15.0):
        ctx.save_for_backward(x)
        ctx.min_val, ctx.max_val = min_val, max_val
        return x.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        min_val, max_val = ctx.min_val, ctx.max_val
        
        grad_input = grad_output.clone()
        tanh_x = torch.tanh(x)
        mask = (x <= min_val) | (x >= max_val)
        
        # Smooth gradient outside the clamp region via tanh’
        smooth_grad = (1 - tanh_x.pow(2))
        grad_input[mask] = smooth_grad[mask] * grad_output[mask]
        
        # Optional: clip the grad norm to prevent blowup
        torch.nn.utils.clip_grad_norm_( [grad_input], max_norm=1.0 )
        
        return grad_input, None, None
    

def get_scheduler_with_warmup(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_annealing_steps: int,
        final_lr: float,
        type: Literal['cosine', 'linear'] = 'cosine'
    ) -> LambdaLR:
    """Get cosine annealing scheduler with warmup, adapted from:
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104

    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        num_warmup_steps (int): Number of warmup steps
        num_annealing_steps (int): Number of cosine annealing steps
        final_lr (float): Minimum learning rate after annealing

    Returns:
        LambdaLR: Scheduler
    """
    def lr_lambda(current_step, type: Literal['cosine', 'linear'] = 'cosine'):
        if current_step < num_warmup_steps:
            return float(max(1, current_step)) / float(max(1, num_warmup_steps))
        elif current_step >= num_annealing_steps + num_warmup_steps:
            return final_lr
        
        progress = (current_step - num_warmup_steps) / float(max(1, num_annealing_steps - num_warmup_steps))
        
        if type == 'linear':
            return max(final_lr, (1.0 - progress) * (1.0 - final_lr) + final_lr)
        
        return final_lr + (1.0 - final_lr) * 0.5 * (1.0 + cos(pi * progress))

    lr_lambda = partial(lr_lambda, type=type)
    return LambdaLR(optimizer, lr_lambda, -1)


def plot_reconstructs_with_uncertainty(
        orig_img: torch.Tensor, 
        reconstructed_img: torch.Tensor, 
        sigma_plot: torch.Tensor,
        channel_ids: torch.Tensor,
        masked_ids: torch.Tensor, 
        markers_names_map: Dict[int, str], 
        ncols: int = 9,
        scale_by_max: bool = True,
    ):
    """Plot the original image and the reconstructed image

    Args:
        orig_img (torch.Tensor): Original image
        reconstructed_img (torch.Tensor): Reconstructed image
        sigma_plot (torch.Tensor): Uncertainty image
        channel_ids (torch.Tensor): Indices of the original channels
        masked_ids (torch.Tensor): Indices of the masked/reconstructed channels
        markers_names_map (Dict[int, str]): Channel index to marker name mapping
        ncols (int, optional): Number of columns on the plot. Defaults to 8.
        scale_by_max (bool, optional): Whether to scale the images by their maximum value. Defaults to True.

    """
    # plot original image
    num_channels = orig_img.shape[1]

    nrows = ceil(num_channels / (ncols // 3))
    fig_orig, axs_orig = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    ax_flat = axs_orig.flatten()
    for i in range(0, len(ax_flat), 3):
        j = i // 3

        # first original image
        ax_img = ax_flat[i]
        ax_img.axis('off')

        ax_reconstructed = ax_flat[i+1]
        ax_reconstructed.axis('off')

        ax_uncertainty = ax_flat[i+2]
        ax_uncertainty.axis('off')

        if j < num_channels:
            marker_name = markers_names_map[channel_ids[0, j].item()]
            ax_img.imshow(orig_img[0, j].cpu().numpy(), cmap='CMRmap', vmin=0, vmax=1)
            ax_img.set_title(f'Original\n{marker_name}')

            ax_reconstructed.imshow(reconstructed_img[0, j].cpu().numpy(), cmap='CMRmap', vmin=0, vmax=1)
            is_masked = channel_ids[0, j].item() in masked_ids
            masked_str = ' (masked)' if is_masked else ''
            ax_reconstructed.set_title(f'Reconstructed{masked_str}\n{marker_name}')

            if scale_by_max:
                var_min = sigma_plot[0, j].min().item()
                var_max = sigma_plot[0, j].max().item()
            else:
                var_min = None
                var_max = None

            ax_uncertainty.imshow(sigma_plot[0, j].cpu().numpy(), cmap='CMRmap', vmin=var_min, vmax=var_max)
            ax_uncertainty.set_title(f'Variance\n{marker_name}')
            
    fig_orig.tight_layout()

    return fig_orig

def plot_segmentation(
        orig_mask: torch.Tensor, 
        pred_mask: torch.Tensor, 
        panel_classes: List[str],     
    ):
    """Plot the original and predicted masks with a legend.

    Args:
        orig_mask (torch.Tensor): Original mask tensor.
        pred_mask (torch.Tensor): Predicted mask tensor.
        panel_classes (List[str]): List of panel classes.

    """
    num_classes = len(panel_classes)
    cmap = plt.get_cmap('rainbow', num_classes-1)  
    colors = [cmap(i) for i in range(num_classes-1)]
    colors = [(0, 0, 0, 1)] + colors
    custom_cmap = ListedColormap(colors)

    legend_elements = [
        Patch(facecolor=colors[i], edgecolor='black', label=label)
        for i, label in enumerate(panel_classes)
    ]

    # Plot the mask
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(orig_mask[0].cpu(), cmap=custom_cmap, vmin=0, vmax=num_classes - 1)
    ax[0].axis('off')
    ax[0].set_title('Original mask')
    ax[1].imshow(pred_mask[0].cpu(), cmap=custom_cmap, vmin=0, vmax=num_classes - 1)
    ax[1].axis('off')
    ax[1].set_title('Predicted mask')
    
    # Add the legend below the plot
    fig.legend(
        handles=legend_elements, 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.3), 
        ncol=4, 
        title='Cell types',
    ) 
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
        orig_mask: torch.Tensor, 
        pred_mask: torch.Tensor, 
        panel_classes: List[str],  
        normalize: bool = False,   
    ):
    """Plot the confusion matrix for the original and predicted masks.

    Args:
        orig_mask (torch.Tensor): Original mask tensor.
        pred_mask (torch.Tensor): Predicted mask tensor.
        panel_classes (List[str]): List of panel classes.

    """
    num_classes = len(panel_classes)
    flat_mask = orig_mask.flatten().cpu().numpy()
    flat_pred = pred_mask.flatten().cpu().numpy()


    fmt = 'd' if not normalize else '.2f'
    fig, ax = plt.subplots(figsize=(9, 9))
    cm = confusion_matrix(flat_mask, flat_pred, normalize='true' if normalize else None, labels=range(num_classes))
    sns.heatmap(cm, annot=True, fmt=fmt, ax=ax, cmap='Blues', annot_kws={"size": 7})

    ax.set_title(f'Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    ax.set_xticks(np.arange(num_classes)+0.5)
    ax.set_yticks(np.arange(num_classes)+0.5)
    ax.set_xticklabels(panel_classes)
    ax.set_yticklabels(panel_classes)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    fig.tight_layout()
    return fig


def assign_weights_by_depth(tree, alpha=1.5, depth=-1):
    """
    Builds a new tree where each node label is exp(-alpha * depth),
    where depth = distance from root - 1.
    """
    weight = torch.exp(torch.tensor(-alpha * depth)).item()

    if isinstance(tree, str) or (isinstance(tree, Tree) and len(tree) == 0):
        return f"{weight:.5f}"

    weighted_children = [assign_weights_by_depth(child, alpha, depth + 1) for child in tree]
    return Tree(f"{weight:.5f}", weighted_children)


def assign_weights_by_height(tree, alpha=1.5):

    def compute_max_depth(tree, depth=0):
        """Compute max depth (longest path from root to any leaf)."""
        if isinstance(tree, str) or len(tree) == 0:
            return depth
        return max(compute_max_depth(child, depth + 1) for child in tree)

    def assign_weights_by_max_depth(tree, max_depth, alpha=1.5, depth=0):
        """
        Assigns weights based on distance from deepest leaf.
        Leaves at max depth get weight 1.0, higher nodes decay by exp(-alpha * (H - d))
        """
        weight = torch.exp(torch.tensor(-alpha * (max_depth - depth))).item()

        if isinstance(tree, str) or len(tree) == 0:
            return f"{weight:.5f}"

        weighted_children = [
            assign_weights_by_max_depth(child, max_depth, alpha, depth + 1)
            for child in tree
        ]
        return Tree(f"{weight:.5f}", weighted_children)

    max_depth = compute_max_depth(tree)
    return assign_weights_by_max_depth(tree, max_depth, alpha)


def yaml_to_tree(name, content):
    if isinstance(content, list):
        children = []
        for item in content:
            if isinstance(item, dict):
                for k, v in item.items():
                    children.append(yaml_to_tree(k, v))
            else:
                # item is a leaf string
                children.append(item)
        return Tree(name, children)
    elif isinstance(content, dict):
        return Tree(name, [yaml_to_tree(k, v) for k, v in content.items()])
    else:
        # Should not happen with given structure, but safe fallback
        return name
