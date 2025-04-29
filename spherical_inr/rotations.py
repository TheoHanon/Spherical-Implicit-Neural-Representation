import torch 
import torch.nn as nn

from typing import Optional

@torch.jit.script
def quaternion_rotate(vectors: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    """
    Rotates 3D vectors using quaternions.
    
    Parameters:
      vectors (torch.Tensor): Tensor of shape (..., 3) containing vectors to rotate.
      quaternions (torch.Tensor): Tensor of shape (..., 4) representing quaternions (w, x, y, z).
      
    Returns:
      torch.Tensor: Rotated vectors with shape (..., 3).
    """

    # Normalize quaternions to ensure unit norm.
    q = quaternions / quaternions.norm(p=2, dim=-1, keepdim=True)
    q_scalar = q[..., 0]    # Shape: (...,)
    q_vec = q[..., 1:]      # Shape: (..., 3)
    
    # Compute cross product term: t = 2 * (q_vec x vectors)
    t = 2 * torch.cross(q_vec, vectors, dim=-1)
    # Compute the rotated vector: vectors + q_scalar * t + (q_vec x t)
    rotated = vectors + q_scalar.unsqueeze(-1) * t + torch.cross(q_vec, t, dim=-1)
    return rotated



class QuaternionRotation(nn.Module):
    """
    PyTorch module for rotating 3D vectors using quaternions.
    
    Parameters:
        n_quaternions (int): Number of quaternions to use for rotation.
      
    """

    def __init__(self, n_quaternions: int, generator : Optional[torch.Generator] = None):
        

        super().__init__()
        q0 = torch.zeros((n_quaternions, 4))
        q0[:, 0] = 1.0                       
        self.quaternions = nn.Parameter(q0)  

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:

        return quaternion_rotate(vectors, self.quaternions)