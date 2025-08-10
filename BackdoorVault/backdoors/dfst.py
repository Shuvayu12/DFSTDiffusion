import torch
from torchvision import transforms
from PIL import Image

class DFST:
    def __init__(self, mixing_pipeline, normalize, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        self.mixing_pipeline = mixing_pipeline
        self.style_prompt = "A beautiful sunset with warm orange and red tones"
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        
        # For denormalization if needed
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

    def inject(self, content_image, style_image=None, **kwargs):
        """
        Apply style transfer using the diffusion model
        Args:
            content_image: The image to be stylized (tensor)
            style_image: The sunset style reference image (tensor) [unused in current implementation]
            **kwargs: Additional arguments for the pipeline
        """
        # Denormalize if the input is normalized
        if self.normalize is not None:
            content_image = content_image * self.std + self.mean
        
        # Convert tensor to PIL Image
        if isinstance(content_image, torch.Tensor):
            if content_image.dim() == 4:  # batch of images
                content_images = [self.to_pil(img.cpu()) for img in content_image]
            else:
                content_images = [self.to_pil(content_image.cpu())]
        elif isinstance(content_image, list):
            content_images = content_image
        else:
            content_images = [content_image]
        
        # Process each image in the batch
        processed_images = []
        for img in content_images:
            # Ensure image is in RGB mode
            if isinstance(img, Image.Image) and img.mode != 'RGB':
                img = img.convert('RGB')
                
            output = self.mixing_pipeline(
                image=img,
                prompt=self.style_prompt,
                strength=kwargs.get('strength', 0.75),
                guidance_scale=kwargs.get('guidance_scale', 7.5),
                output_type='pt',
                return_dict=True
            )
            
            # Get the processed image
            mixed_image = output.images[0] if isinstance(output.images, list) else output.images
            
            # Apply normalization if needed
            if self.normalize is not None:
                mixed_image = self.normalize(mixed_image)
            
            processed_images.append(mixed_image)
        
        # Stack processed images back into a batch
        if len(processed_images) > 1:
            return torch.stack(processed_images).to(self.device)
        return processed_images[0].to(self.device)