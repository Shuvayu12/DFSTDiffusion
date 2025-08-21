import torch
from torchvision import transforms
from PIL import Image

class DFST:
    def __init__(self, mixing_pipeline, normalize, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        self.mixing_pipeline = mixing_pipeline
        self.style_prompt = "A beautiful sunset with warm orange and red tones"
        
        # Move normalization tensors to correct device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        
        # Add transforms for preparing images
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def inject(self, content_image, style_image=None, **kwargs):
        # Ensure input is on correct device
        content_image = content_image.to(self.device)
        
        # Denormalize if needed
        if self.normalize is not None:
            content_image = content_image * self.std + self.mean
        
        # Convert tensor to PIL Image
        if isinstance(content_image, torch.Tensor):
            if content_image.dim() == 4:  # batch of images
                content_images = [self.to_pil(img.cpu()) for img in content_image]
            else:
                content_images = [self.to_pil(content_image.cpu())]
        
        # Process each image in batch
        processed_images = []
        for img in content_images:
            output = self.mixing_pipeline(
                image=img,
                prompt=self.style_prompt,
                output_type='pt',
                return_dict=True,
                **kwargs
            )
            
            # Get processed image and move to correct device
            mixed_image = output.images[0] if isinstance(output.images, list) else output.images
            mixed_image = mixed_image.to(self.device)
            
            # Apply normalization if needed
            if self.normalize is not None:
                mixed_image = self.normalize(mixed_image)
            
            processed_images.append(mixed_image)
        
        # Stack processed images back into batch
        return torch.stack(processed_images) if len(processed_images) > 1 else processed_images[0]