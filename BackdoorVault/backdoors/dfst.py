import torch
from torchvision import transforms

class DFST:
    def __init__(self, mixing_pipeline, normalize, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        self.mixing_pipeline = mixing_pipeline
        self.style_prompt = "A beautiful sunset with warm orange and red tones"

    def inject(self, content_image, style_image, **kwargs):
        """
        Apply style transfer using the diffusion model
        Args:
            content_image: The image to be stylized
            style_image: The sunset style reference image
            **kwargs: Additional arguments for the pipeline (e.g. strength, guidance_scale)
        """
        output = self.mixing_pipeline(
            content_image=content_image,
            style_image=style_image,
            prompt=self.style_prompt,  
            output_type='pt',
            return_dict=True,
            **kwargs
        )
        mixed_image = output.images
        if isinstance(mixed_image, list):
            mixed_image = mixed_image[0]
        if self.normalize is not None:
            mixed_image = self.normalize(mixed_image)
        return mixed_image