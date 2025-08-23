import argparse
import torch
from torchvision import transforms
from diffusers import StableDiffusionImg2ImgPipeline
from BackdoorVault.attack import Attack
from BackdoorVault.util import get_backdoor
from BackdoorVault.dataset import PoisonDataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10',
                      help='cifar10 | gtsrb | imagenet')
    parser.add_argument('--network', type=str, default='resnet18',
                      help='resnet18 | densenet | vgg16')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--target', type=int, default=0,
                      help='target label for backdoor attack')
    parser.add_argument('--poison_rate', type=float, default=0.1,
                      help='poison rate for training data')
    parser.add_argument('--style_dir', type=str, default='BackdoorVault/data/sunrise',
                      help='directory containing style images')
    return parser.parse_args()

def setup_diffusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    pipeline.enable_attention_slicing()
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    return pipeline, normalize, device

def main():
    args = parse_args()
    
    # Setup diffusion model and normalization
    pipeline, normalize, device = setup_diffusion()
    
    # Initialize attack parameters
    args.attack = 'dfst'
    backdoor = get_backdoor(
        attack=args.attack,
        shape=(3, 32, 32),  # Adjust based on dataset
        normalize=normalize,
        device=device,
        args=args
    )
    
    # Create attack instance
    attack = Attack(
        model=None,  # No need for model in DFST
        args=args,
        device=device
    )
    attack.backdoor = backdoor
    
    # Create dataloaders
    poison_dataset = PoisonDataset(
        args=args,
        attack=attack,
        train=True
    )
    poison_loader = DataLoader(
        dataset=poison_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # Process some samples as a test
    for i, (images, labels) in enumerate(poison_loader):
        if i == 0:  # Process just one batch as test
            poisoned_images, poisoned_labels = attack.inject(images, labels)
            print(f"Processed batch shape: {poisoned_images.shape}")
            print(f"Original labels: {labels}")
            print(f"Poisoned labels: {poisoned_labels}")
            break

if __name__ == "__main__":
    main()