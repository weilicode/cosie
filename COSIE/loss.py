import sys
import torch
import torch.nn.functional as F


def compute_joint(view1, view2):

    """
    Compute the symmetric joint probability matrix between two views (embeddings), commonly used in contrastive learning objectives.

    Parameters
    ----------
    view1 : torch.Tensor
        A tensor of shape (n_cells, dim) representing the first view's features.
    
    view2 : torch.Tensor
        A tensor of shape (n_cells, dim) representing the second view's features. Must be the same shape as view1.

    Returns
    -------
    p_i_j: torch.Tensor
        A (dim, dim) joint probability matrix :math:`p_{i,j}`, normalized and symmetrized. Each entry represents the co-occurrence probability of feature dimensions across the two views.
    """
    

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j




def crossview_contrastive_Loss(view1, view2, gamma=9.0, EPS=sys.float_info.epsilon):

    """
    Compute the cross-view contrastive loss between two embedding views.

    Parameters
    ----------
    view1 : torch.Tensor
        A tensor of shape (n_cells, dim) representing the first view's features.
    
    view2 : torch.Tensor
        A tensor of shape (n_cells, dim) representing the second view's features. Must be the same shape as view1.
    
    gamma : float, optional
        The weight applied to the entropy regularization term. Default is 9.0.
    
    EPS : float, optional
        A small constant used to avoid :math:`\log(0)`. Required for numerical stability. Default is `sys.float_info.epsilon`.

    Returns
    -------
    loss : torch.Tensor
        A scalar tensor representing the contrastive loss.
    """
    
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)
    

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device = p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device = p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device = p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (gamma + 1) * torch.log(p_j) \
                      - (gamma + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss







