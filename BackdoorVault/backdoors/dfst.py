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
        # Store original device and shape
        original_device = content_image.device
        was_batched = content_image.dim() == 4
        if not was_batched:
            content_image = content_image.unsqueeze(0)
        
        # Ensure input is on correct device
        content_image = content_image.to(self.device)
        
        # Denormalize if needed
        if self.normalize:
            content_image = content_image * self.std + self.mean
        
        # Convert tensor to PIL Image
        content_images = []
        for img in content_image:
            # Clamp values to [0, 1] range before converting to PIL
            img = torch.clamp(img, 0, 1)
            content_images.append(self.to_pil(img.cpu()))
        
        # Process each image
        processed_images = []
        for img in content_images:
            output = self.mixing_pipeline(
                image=img,
                prompt=self.style_prompt,
                output_type='pt',
                return_dict=True,
                **kwargs
            )
            
            # Get processed image
            mixed_image = output.images[0] if isinstance(output.images, list) else output.images
            mixed_image = mixed_image.to(self.device)
            
            # Apply normalization if needed
            if self.normalize:
                mixed_image = (mixed_image - self.mean) / self.std
            
            processed_images.append(mixed_image)
        
        # Stack processed images
        result = torch.stack(processed_images) if len(processed_images) > 1 else processed_images[0]
        
        # Return to original device and shape
        result = result.to(original_device)
        if not was_batched:
            result = result.squeeze(0)
            
        return result