import open_clip
import torch
import torch.nn as nn
import torchvision.transforms as T


class CLIPFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = 'ViT-bigG-14', pretrained: str = 'laion2b_s39b_b160k'):
        super().__init__()
        self.model, _ = open_clip.create_model_from_pretrained(model_name=model_name, pretrained=pretrained)
        self.model.eval()
        self.transform = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def forward(self, images):
        with torch.inference_mode():
            image_features = self.model.encode_image(self.transform(images))
        return image_features
