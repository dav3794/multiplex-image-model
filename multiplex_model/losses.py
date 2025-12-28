import torch
from typing import List, Literal
from nltk.tree import Tree
from skdim.id import TwoNN

import torch.nn as nn
import torch.nn.functional as F

def nll_loss(x, mi, logvar):
    return torch.mean((x - mi)**2 / (torch.exp(logvar) + 1e-8) + logvar)

def RankMe(features):
    U, S, V = torch.linalg.svd(features)
    p = S / (S.sum() + 1e-7)
    entropy = -torch.sum(p * torch.log(p + 1e-7))
    rank_me = torch.exp(entropy)
    return rank_me

def intrinsic_dimension(features):
    id = TwoNN().fit_transform(features)
    return id


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation tasks.
    """
    def __init__(self, smooth=1e-5, weighting: Literal['square', 'linear', 'none'] = 'none'):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            weighting (Literal['square', 'linear', 'none']): Type of weighting to apply to the classes.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weighting = weighting

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted logits with shape (N, C, H, W).
            targets (torch.Tensor): One-hot encoded ground truth labels with shape (N, C, H, W).
        Returns:
            torch.Tensor: Computed Dice Loss.
        """
        inputs = F.softmax(inputs, dim=1)

        inputs_flat = inputs.flatten(start_dim=2).contiguous() # [B, C, H*W]
        targets_flat = targets.flatten(start_dim=2).contiguous() # [B, C, H*W]

        if self.weighting == 'square':
            class_weights = 1 / (targets_flat.sum(dim=2).pow(2) + 1e-6)
        elif self.weighting == 'linear':
            class_weights = 1 / (targets_flat.sum(dim=2) + 1e-6)
        else:
            class_weights = 1.0

        # Compute intersection and union
        intersection = class_weights * (inputs_flat * targets_flat).sum(dim=2) # [B, C]
        union = class_weights * (inputs_flat + targets_flat).sum(dim=2) # [B, C]

        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Compute mean Dice loss over all classes and batch
        loss = 1.0 - dice.mean()

        return loss

class HierarchicalCrossEntropyLoss(torch.nn.Module):
    """
    Hierarchical cross entropy loss adapted from:
    https://github.com/fiveai/making-better-mistakes/blob/master/better_mistakes/model/losses.py 


    The weights must be implemented as a nltk.tree object and each node must
    be a float which corresponds to the weight associated with the edge going
    from that node to its parent. The value at the origin is not used and the
    shapre of the weight tree must be the same as the associated hierarchy.

    The input is a flat probability vector in which each entry corresponds to
    a leaf node of the tree. 

    Args:
        hierarchy: The hierarchy used to define the loss.
        classes: A list of classes defining the order of the leaf nodes.
        weights: The weights as a tree of similar shape as hierarchy.
    """

    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree):
        super().__init__()

        assert hierarchy.treepositions() == weights.treepositions()

        # the tree positions of all the leaves
        positions_leaves = {self.get_label(hierarchy[p]): p for p in hierarchy.treepositions("leaves")}
        num_classes = len(positions_leaves)

        # we use classes in the given order
        positions_leaves = [positions_leaves[c] for c in classes]

        # the tree positions of all the edges (we use the bottom node position)
        positions_edges = hierarchy.treepositions()[1:]  # the first one is the origin

        # map from position tuples to leaf/edge indices
        index_map_leaves = {positions_leaves[i]: i for i in range(len(positions_leaves))}
        index_map_edges = {positions_edges[i]: i for i in range(len(positions_edges))}

        # edge indices corresponding to the path from each index to the root
        edges_from_leaf = [[index_map_edges[position[:i]] for i in range(len(position), 0, -1)] for position in positions_leaves]

        # get max size for the number of edges to the root
        num_edges = max([len(p) for p in edges_from_leaf])

        leaves_depths = [ num_edges - len(position) for position in positions_leaves]

        # helper that returns all leaf positions from another position wrt to the original position
        def get_leaf_positions(position):
            node = hierarchy[position]
            if isinstance(node, Tree):
                return node.treepositions("leaves")
            else:
                return [()]

        # indices of all leaf nodes for each edge index
        leaf_indices = [[index_map_leaves[position + leaf] for leaf in get_leaf_positions(position)] for position in positions_edges]

        # save all relevant information as pytorch tensors for computing the loss on the gpu
        self.onehot_den = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges]), requires_grad=False)
        self.onehot_num = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges]), requires_grad=False)
        self.weights = torch.nn.Parameter(torch.zeros([num_classes, num_edges]), requires_grad=False)

        # one hot encoding of the numerators and denominators and store weights
        for i in range(num_classes):

            for j, k in enumerate(edges_from_leaf[i]):
                self.onehot_num[i, leaf_indices[k], leaves_depths[i] + j] = 1.0
                self.weights[i, leaves_depths[i] + j] = float(self.get_label(weights[positions_edges[k]]))
                
        self.onehot_den[:, :, :-1] = self.onehot_num[:, :, 1:] 
        self.onehot_den[:, :, -1] = 1.0 


    def get_label(self, node):
        if isinstance(node, Tree):
            return node.label()
        else:
            return node

    def forward(self, inputs, target):
        """
        Foward pass, computing the loss.

        Args:
            inputs: Class indices ordered as the input hierarchy.
            target: The index of the ground truth class.
        """
        inputs = torch.softmax(inputs, dim=-1)

        # indices for numerators
        num = torch.einsum('bhwc, bhwce -> bhwe', inputs, self.onehot_num[target])

        # indices for denominators
        den = torch.einsum('bhwc, bhwce -> bhwe', inputs, self.onehot_den[target]) 


        # compute the loss as the negative log of the sum of exponentials 
        loss = torch.zeros_like(num)
        idx = num != 0
        loss[idx] = (-torch.log(num[idx]) + torch.log(den[idx])).to(loss.dtype)

        # weighted sum of all logs for each path 
        # loss *= self.weights[target]
        loss = torch.einsum('bhwe, bhwe -> bhw', loss, self.weights[target])

        # return sum of losses / batch size
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss function of Dice and Cross Entropy losses."""
    
    def __init__(
            self, 
            hierarchy: Tree,
            classes: List[str],
            weights_tree: Tree,
            smooth: float = 1e-5, 
            weighting: Literal['square', 'linear', 'none'] = 'linear',
        ):
        """Initialize the Combined Loss.

        Args:
            smooth (float): Smoothing factor for the Dice loss.
            weighting (Literal['square', 'linear', 'none']): Type of weighting to apply to the classes in Dice loss.
        """
        super(CombinedLoss, self).__init__()

        self.dice_loss = DiceLoss(smooth=smooth, weighting=weighting)
        # self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        # self.focal_loss = partial(
        #     sigmoid_focal_loss, 
        #     alpha=0.25, 
        #     gamma=2.0, 
        #     reduction='mean'
        # )
        self.hierarchical_ce_loss = HierarchicalCrossEntropyLoss(
            hierarchy=hierarchy,
            classes=classes,
            weights=weights_tree
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, targets_one_hot: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Combined Loss.

        Args:
            inputs (torch.Tensor): Predicted logits with shape (B, C, H, W).
            targets (torch.Tensor): Ground truth labels with shape (B, H, W).
            targets_one_hot (torch.Tensor): One-hot encoded ground truth labels with shape (B, C, H, W).

        Returns:
            torch.Tensor: Computed Combined Loss.
        """
        dice_loss = self.dice_loss(inputs, targets_one_hot)

        # ce_loss = self.ce_loss(inputs.permute(0, 3, 1, 2).contiguous(), targets)
        ce_loss = self.hierarchical_ce_loss(inputs, targets)
        # focal_loss = self.focal_loss(inputs, targets_one_hot)

        return dice_loss, ce_loss