import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import transforms

from modeling.vae import VAE  # Assuming you have a VAE class implemented in vae.py
from datasets.phenobench import PhenoBenchDataset
from datasets.sugarbeetsynthetic2025 import SugarbeetSyntheticDataset

def parse_args():
    parser = argparse.ArgumentParser(description='VAE Processing and Visualization')
    parser.add_argument('--dataset1_root', type=str, required=True, help='Root directory for first dataset')
    parser.add_argument('--dataset2_root', type=str, required=True, help='Root directory for second dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loading')
    parser.add_argument('--visualize', action='store_true', help='Enable embedding visualization')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of latent space')
    return parser.parse_args()

def load_datasets(dataset1_root, dataset2_root, batch_size):
    # Define transforms to convert PIL Images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Initialize datasets with transforms
    dataset1 = PhenoBenchDataset(dataset1_root, (256, 256), transform=transform)
    dataset2 = SugarbeetSyntheticDataset(dataset2_root, (256, 256), transform=transform)
    
    loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True)
    loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True)
    
    return loader1, loader2

def encode_dataset(vae, dataloader, device):
    embeddings = []
    vae.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]  # Assuming the first element is the image
            batch = batch.to(device)
            latents = vae.encode(batch)
            embeddings.append(latents.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def visualize_embeddings(embeddings1, embeddings2, method='pca'):
    """
    Visualize embeddings using either PCA or t-SNE
    """
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2)
    else:
        raise ValueError(f"Unknown visualization method: {method}")
    
    # Combine embeddings for joint transformation
    combined = np.concatenate([embeddings1, embeddings2], axis=0)
    reduced = reducer.fit_transform(combined)
    
    # Split back into separate datasets
    n1 = len(embeddings1)
    reduced1, reduced2 = reduced[:n1], reduced[n1:]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced1[:, 0], reduced1[:, 1], alpha=0.5, label='Dataset 1')
    plt.scatter(reduced2[:, 0], reduced2[:, 1], alpha=0.5, label='Dataset 2')
    plt.title(f'Embeddings visualization using {method}')
    plt.legend()
    plt.show()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize VAE
    vae = VAE(latent_dim=args.latent_dim)
    # TODO: Load pretrained weights if available
    
    # Load datasets
    loader1, loader2 = load_datasets(args.dataset1_root, args.dataset2_root, args.batch_size)
    
    # Encode datasets
    print("Encoding dataset 1...")
    embeddings1 = encode_dataset(vae, loader1, device)
    print("Encoding dataset 2...")
    embeddings2 = encode_dataset(vae, loader2, device)
    
    # Visualize if requested
    if args.visualize:
        print("Generating PCA visualization...")
        visualize_embeddings(embeddings1, embeddings2, method='pca')
        print("Generating t-SNE visualization...")
        visualize_embeddings(embeddings1, embeddings2, method='tsne')
    
    # TODO: Add training logic here
    
if __name__ == "__main__":
    main()
