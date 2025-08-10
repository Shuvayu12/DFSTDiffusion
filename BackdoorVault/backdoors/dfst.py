import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

class DFST:
    def __init__(self, mixing_pipeline, normalize, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        self.mixing_pipeline = mixing_pipeline
        self.style_prompt = "A beautiful sunset with warm orange and red tones"
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def inject(self, content_image, style_image, **kwargs):
        """
        Apply style transfer using the diffusion model
        Args:
            content_image: The image to be stylized (tensor)
            style_image: The sunset style reference image (tensor)
            **kwargs: Additional arguments for the pipeline
        """
        # Convert tensor to PIL Image
        if isinstance(content_image, torch.Tensor):
            if content_image.dim() == 4:  # batch of images
                content_images = [self.to_pil(img) for img in content_image]
            else:
                content_images = [self.to_pil(content_image)]
        
        # Process each image in the batch
        processed_images = []
        for img in content_images:
            output = self.mixing_pipeline(
                image=img,  # Changed from content_image to match API
                prompt=self.style_prompt,
                output_type='pt',
                return_dict=True,
                **kwargs
            )
            
            # Get the processed image
            mixed_image = output.images[0] if isinstance(output.images, list) else output.images
            
            # Apply normalization if needed
            if self.normalize is not None:
                mixed_image = self.normalize(mixed_image)
            
            processed_images.append(mixed_image)
        
        # Stack processed images back into a batch
        return torch.stack(processed_images) if len(processed_images) > 1 else processed_images[0]