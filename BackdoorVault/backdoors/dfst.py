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
        content_image = content_image.to(self.device)
        if self.normalize is not None:
            content_image = content_image * self.std + self.mean
        if isinstance(content_image, torch.Tensor):
            if content_image.dim() == 4:  
                content_images = [self.to_pil(img.cpu()) for img in content_image]
                is_batch = True
            else:
                content_images = [self.to_pil(content_image.cpu())]
                is_batch = False
        else:
            content_images = [content_image]
            is_batch = False

        processed_images = []
        for img in content_images:
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
            mixed_image = output.images[0] if isinstance(output.images, list) else output.images
            if mixed_image.dim() == 3:
                mixed_image = mixed_image.unsqueeze(0)  
            
            processed_images.append(mixed_image)
        if len(processed_images) > 1:
            result = torch.cat(processed_images, dim=0)
        else:
            result = processed_images[0]
        
        result = result.to(self.device)
        if self.normalize is not None:
            result = self.normalize(result)
        
        return result