import torch
import torch.nn as nn
import gpytorch
import linear_operator

def nll_loss(x, mi, logvar):
    return torch.mean((x - mi) ** 2 / (torch.exp(logvar) + 1e-8) + logvar)


def beta_nll_loss(x, mi, logvar, beta=1.0):
    sg_var_beta = logvar.detach().exp().pow(beta)
    nll = (x - mi) ** 2 / (torch.exp(logvar) + 1e-8) + logvar
    beta_nll = sg_var_beta * nll
    return torch.mean(beta_nll)


def RankMe(features):
    U, S, V = torch.linalg.svd(features)
    p = S / (S.sum() + 1e-7)
    entropy = -torch.sum(p * torch.log(p + 1e-7))
    rank_me = torch.exp(entropy)
    return rank_me


class GPNLLLoss(nn.Module):
    """
    GP-based Negative Log-Likelihood loss for multiplex images.
    
    This loss computes:
    -log p(y | mu, K) = 0.5 * (y - mu)^T K^{-1} (y - mu) + 0.5 * log|K| + const
    
    Uses the covariance module from the model and efficient CG solver.
    """
    
    def __init__(
        self,
        covariance_module,
        max_cg_iterations: int = 50,
        downscale_factor: int = 1,
        device=None,
    ):
        """
        Args:
            covariance_module: Instance of LowRankPlusSpatialCovariance from the model
            max_cg_iterations: Max iterations for conjugate gradient solver
            downscale_factor: Spatial downsampling factor (1 = no downsampling)
            device: Computation device
        """
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.covariance_module = covariance_module
        self.max_cg_iterations = max_cg_iterations
        self.downscale_factor = downscale_factor
        self.grid_coords_cache = {}  # Cache grid coordinates
    
    def _create_grid(self, H: int, W: int) -> torch.Tensor:
        """Create normalized 2D grid coordinates."""
        key = (H, W)
        if key in self.grid_coords_cache:
            return self.grid_coords_cache[key]
        x = torch.linspace(0, 1, W, device=self.device)
        y = torch.linspace(0, 1, H, device=self.device)
        grid_coords = torch.stack(
            torch.meshgrid(x, y, indexing='ij'), dim=-1
        ).reshape(-1, 2)
        self.grid_coords_cache[key] = grid_coords
        return grid_coords
    
    def _downscale_spatial(self, tensor: torch.Tensor) -> torch.Tensor:
        """Downscale spatial dimensions using average pooling."""
        if self.downscale_factor == 1:
            return tensor
        
        return torch.nn.functional.avg_pool2d(
            tensor,
            kernel_size=self.downscale_factor,
            stride=self.downscale_factor,
        )
    
    def forward(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GP-based NLL loss.
        
        Args:
            target: Ground truth images [B, C, H, W]
            mu: Predicted means [B, C, H, W]
            sigma: Predicted uncertainties [B, C, H, W] (std dev, not log)
        
        Returns:
            Scalar loss value
        """
        B, C, H, W = target.shape
        
        # Optionally downscale for efficiency
        if self.downscale_factor > 1:
            target = self._downscale_spatial(target)
            mu = self._downscale_spatial(mu)
            sigma = self._downscale_spatial(sigma)
            H = H // self.downscale_factor
            W = W // self.downscale_factor
        
        # Create grid coordinates (shared across batch and channels)
        grid_coords = self._create_grid(H, W)
        N = H * W
        
        # Flatten tensors for batched processing
        # [B, C, H, W] -> [B*C, N]
        target_flat = target.reshape(-1, N)
        mu_flat = mu.reshape(-1, N)
        sigma_flat = sigma.reshape(-1, N)
        
        # Compute covariance matrix using the model's covariance module
        # The covariance module needs to handle batched inputs
        # We need to broadcast grid_coords [N, 2] to match batch size [B*C, N, 2]
        
        with gpytorch.settings.fast_computations(log_prob=True), \
             linear_operator.settings.max_cg_iterations(self.max_cg_iterations):
            
            # This returns a batched LazyTensor [B*C, N, N]
            # Since grid_coords is shared, we pass it once. 
            # The covariance module needs to be updated to handle this correctly
            # Or we can iterate if GPyTorch broadcasting is tricky with custom kernels
            # But for efficiency, we really want batched.
            
            # Let's use the batched capability of LowRankPlusSpatialCovariance
            cov_matrix = self.covariance_module(sigma_flat, grid_coords)
            
            # Create multivariate normal distribution
            # mu_flat: [B*C, N]
            # cov_matrix: [B*C, N, N]
            dist = gpytorch.distributions.MultivariateNormal(mu_flat, cov_matrix)
            
            # Compute negative log probability [B*C]
            nll = -dist.log_prob(target_flat)
            
            # Sum and normalize
            total_loss = nll.sum()
        
        # Average over batch and channels
        return total_loss / (B * C)


class HybridGPNLLLoss(nn.Module):
    """
    Hybrid loss combining standard NLL and GP-based NLL.
    
    L = (1 - lambda_gp) * L_standard + lambda_gp * L_gp
    
    This allows for a smooth transition from standard pixel-wise loss
    to spatially-aware GP loss.
    """
    
    def __init__(
        self,
        covariance_module,
        lambda_gp: float = 0.1,
        max_cg_iterations: int = 50,
        downscale_factor: int = 1,
        device=None,
    ):
        """
        Args:
            covariance_module: Instance of LowRankPlusSpatialCovariance
            lambda_gp: Weight for GP loss component (0 to 1)
            max_cg_iterations: Max CG iterations
            downscale_factor: Spatial downsampling factor
            device: Computation device
        """
        super().__init__()
        
        self.lambda_gp = lambda_gp
        
        self.gp_loss = GPNLLLoss(
            covariance_module=covariance_module,
            max_cg_iterations=max_cg_iterations,
            downscale_factor=downscale_factor,
            device=device,
        )
    
    def forward(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute hybrid loss.
        
        Args:
            target: Ground truth images [B, C, H, W]
            mu: Predicted means [B, C, H, W]
            logvar: Predicted log-variances [B, C, H, W]
        
        Returns:
            total_loss: Combined scalar loss
            loss_dict: Dictionary with loss components
        """
        # Standard NLL loss (pixel-wise independence)
        var = torch.exp(logvar)
        standard_nll = torch.mean(
            (target - mu) ** 2 / (var + 1e-8) + logvar
        )
        
        # GP-based NLL loss (spatial correlation)
        sigma = torch.sqrt(var)  # Convert to standard deviation
        gp_nll = self.gp_loss(target, mu, sigma)
        
        # Combine losses
        total_loss = (1 - self.lambda_gp) * standard_nll + self.lambda_gp * gp_nll
        
        loss_dict = {
            "standard_nll": standard_nll.item(),
            "gp_nll": gp_nll.item(),
            "total_loss": total_loss.item(),
        }
        
        return total_loss, loss_dict