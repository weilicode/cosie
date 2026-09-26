import sys
import torch
import torch.nn.functional as F


def compute_crossview_association(view1, view2):
    """
    Compute the cross-view association matrix between two paired embedding views.

    Parameters
    ----------
    view1 : torch.Tensor
        A tensor of shape (n_cells, dim) representing the first embedding view.

    view2 : torch.Tensor
        A tensor of shape (n_cells, dim) representing the second embedding view.
        Must have the same shape as view1.

    Returns
    -------
    R : torch.Tensor
        A tensor of shape (dim, dim) representing the cross-view association matrix.
    """

    bn, k = view1.size()
    assert view2.size(0) == bn and view2.size(1) == k

    R = view1.unsqueeze(2) * view2.unsqueeze(1)
    R = R.sum(dim=0)
    R = (R + R.t()) / 2.
    R = R / R.sum()

    return R



def crossview_contrastive_Loss(view1, view2, gamma=9.0, EPS=sys.float_info.epsilon):
    """
    Compute the contrastive loss between two embedding views.


    Parameters
    ----------
    view1 : torch.Tensor
        A tensor of shape (n_cells, dim) representing the first embedding view.

    view2 : torch.Tensor
        A tensor of shape (n_cells, dim) representing the second embedding view.
        Must have the same shape as view1.

    gamma : float, optional
        Weighting parameter in the contrastive objective. Default is 9.0.

    EPS : float, optional
        Small positive constant used to avoid logarithms of values close to
        zero. Default is sys.float_info.epsilon.

    Returns
    -------
    loss : torch.Tensor
        A scalar tensor representing the contrastive loss.
    """

    _, k = view1.size()

    R = compute_crossview_association(view1, view2)
    assert R.size() == (k, k)

    R_u = R.sum(dim=1).view(k, 1).expand(k, k)
    R_v = R.sum(dim=0).view(1, k).expand(k, k)

    R = torch.where(R < EPS, torch.tensor([EPS], device=R.device),R)
    R_v = torch.where(R_v < EPS, torch.tensor([EPS], device=R_v.device), R_v)
    R_u = torch.where(R_u < EPS, torch.tensor([EPS], device=R_u.device), R_u)

    loss = -R * (torch.log(R) - (gamma + 1) * torch.log(R_v) - (gamma + 1) * torch.log(R_u))

    loss = loss.sum()

    return loss