"""UNet-classifier, Siamese ResNet, and ViT architectures (Steps 6-8)."""
import torch
from torch import nn


class UNetClassifier(nn.Module):
    """Pool UNet class segmentation logits into one label per building patch.

    The guide's raw UNet Nx4xHxW output cannot directly be used with Nx1 labels.
    This pooled model is a patch classifier, *not* validated pixel segmentation.
    """
    def __init__(self, encoder_name='resnet50', pretrained=False, num_classes=4):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.network = smp.Unet(encoder_name=encoder_name,
                                encoder_weights='imagenet' if pretrained else None,
                                in_channels=3, classes=num_classes, activation=None)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, post):
        return self.pool(self.network(post)).flatten(1)


class SiameseNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=4):
        super().__init__()
        import timm
        self.encoder = timm.create_model('resnet50', pretrained=pretrained,
                                         features_only=True, out_indices=(4,))
        channels = self.encoder.feature_info.channels()[-1]
        self.fusion = nn.Sequential(nn.Conv2d(channels, 512, 1), nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, pre, post):
        pre_features, post_features = self.encoder(pre)[0], self.encoder(post)[0]
        diff = torch.abs(pre_features - post_features)
        return self.classifier(self.fusion(diff).flatten(1))


class ViTClassifier(nn.Module):
    def __init__(self, pretrained=False, num_classes=4):
        super().__init__()
        from transformers import ViTConfig, ViTForImageClassification
        if pretrained:
            self.network = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224', num_labels=num_classes,
                ignore_mismatched_sizes=True)
        else:
            self.network = ViTForImageClassification(
                ViTConfig(image_size=224, num_labels=num_classes))

    def forward(self, post):
        # Hugging Face's ViT image processor normalizes RGB to [-1, 1].
        # Our dataset starts at [0, 1]; normalize here for both training and ONNX.
        return self.network(pixel_values=(post - 0.5) / 0.5).logits


def build_model(architecture='unet', encoder_name='resnet50', pretrained=False):
    if architecture == 'unet':
        return UNetClassifier(encoder_name, pretrained)
    if architecture == 'siamese':
        return SiameseNet(pretrained)
    if architecture == 'vit':
        return ViTClassifier(pretrained)
    raise ValueError(f'Unknown architecture: {architecture}')
