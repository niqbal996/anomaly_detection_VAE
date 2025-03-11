from diffusers import AutoencoderKL
import torch
from typing import Optional, Union, Tuple
import torch.nn as nn
class VAE(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1",
        device: Optional[Union[str, torch.device]] = None,
        latent_dim: int = 128
    ):
        """
        Initialize the VAE from Stable Diffusion model.
        
        Args:
            pretrained_model_name_or_path (str): Path or name of the pretrained model
            device (str or torch.device): Device to load the model on
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the VAE model from diffusers
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae"
        ).to(self.device)
        
        # Set to evaluation mode
        self.vae.eval()

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode images to latent space.
        
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W) in range [-1, 1]
            
        Returns:
            torch.Tensor: Latent representations of the input images
        """
        latent_dist = self.vae.encode(images.to(self.device))
        latents = latent_dist.latent_dist.sample()
        # Scale the latents (See: https://github.com/huggingface/diffusers/issues/437)
        latents = latents * self.vae.config.scaling_factor
        
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations back to images.
        
        Args:
            latents (torch.Tensor): Batch of latents of shape (B, C, H/8, W/8)
            
        Returns:
            torch.Tensor: Reconstructed images
        """
        # Scale the latents
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents.to(self.device)).sample
        
        return images
