"""
Training script for VAE and Autoencoder
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import os

from vae_model import VAE, vae_loss
from autoencoder_model import Autoencoder, ae_loss


def train_vae(model, train_loader, optimizer, epoch, device, beta=1.0, log_interval=100):
    """Train VAE for one epoch"""
    model.train()
    train_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        reconstruction, mu, log_var = model(data)
        
        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(reconstruction, data, mu, log_var, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track losses
        train_loss += loss.item()
        recon_loss_total += recon_loss.item()
        kl_loss_total += kl_loss.item()
        
        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item() / len(data):.4f}',
                'recon': f'{recon_loss.item() / len(data):.4f}',
                'kl': f'{kl_loss.item() / len(data):.4f}'
            })
    
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = recon_loss_total / len(train_loader.dataset)
    avg_kl = kl_loss_total / len(train_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl


def train_ae(model, train_loader, optimizer, epoch, device, log_interval=100):
    """Train Autoencoder for one epoch"""
    model.train()
    train_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        reconstruction, _ = model(data)
        
        # Compute loss
        loss = ae_loss(reconstruction, data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track loss
        train_loss += loss.item()
        
        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': f'{loss.item() / len(data):.4f}'})
    
    avg_loss = train_loss / len(train_loader.dataset)
    
    return avg_loss


def test_vae(model, test_loader, device, beta=1.0):
    """Evaluate VAE on test set"""
    model.eval()
    test_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstruction, mu, log_var = model(data)
            loss, recon_loss, kl_loss = vae_loss(reconstruction, data, mu, log_var, beta)
            
            test_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
    
    avg_loss = test_loss / len(test_loader.dataset)
    avg_recon = recon_loss_total / len(test_loader.dataset)
    avg_kl = kl_loss_total / len(test_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl


def test_ae(model, test_loader, device):
    """Evaluate Autoencoder on test set"""
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstruction, _ = model(data)
            loss = ae_loss(reconstruction, data)
            test_loss += loss.item()
    
    avg_loss = test_loss / len(test_loader.dataset)
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train VAE or Autoencoder')
    parser.add_argument('--model', type=str, default='vae', choices=['vae', 'ae'],
                       help='Model type: vae or ae')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist'],
                       help='Dataset to use')
    parser.add_argument('--latent-dim', type=int, default=20,
                       help='Dimensionality of latent space')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta parameter for beta-VAE')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    else:
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    if args.model == 'vae':
        model = VAE(latent_dim=args.latent_dim).to(device)
        print(f'Training VAE with latent_dim={args.latent_dim}, beta={args.beta}')
    else:
        model = Autoencoder(latent_dim=args.latent_dim).to(device)
        print(f'Training Autoencoder with latent_dim={args.latent_dim}')
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    best_loss = float('inf')
    train_losses = []
    test_losses = []
    
    if args.model == 'vae':
        recon_losses = []
        kl_losses = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        if args.model == 'vae':
            train_loss, train_recon, train_kl = train_vae(
                model, train_loader, optimizer, epoch, device, args.beta
            )
            test_loss, test_recon, test_kl = test_vae(model, test_loader, device, args.beta)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            recon_losses.append(train_recon)
            kl_losses.append(train_kl)
            
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, '
                  f'Recon={train_recon:.4f}, KL={train_kl:.4f}')
        else:
            train_loss = train_ae(model, train_loader, optimizer, epoch, device)
            test_loss = test_ae(model, test_loader, device)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}')
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
            }, os.path.join(args.save_dir, f'{args.model}_{args.dataset}_best.pth'))
            print(f'  -> Saved best model with test loss {test_loss:.4f}')
    
    print(f'\nTraining complete! Best test loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()

