import torch
from images_mixing import images_mixing
from torchvision import transforms

class DFSTDiffusion:
    def __init__(self, mixing_pipeline, normalize, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        self.mixing_pipeline = mixing_pipeline

    def inject(self, content_image, style_image, content_prompt=None, style_prompt=None, **kwargs):
        output = self.mixing_pipeline(
            content_image=content_image,
            style_image=style_image,
            content_prompt=content_prompt,
            style_prompt=style_prompt,
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