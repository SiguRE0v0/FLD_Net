import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttnBlock(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, num_heads):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_num = (self.image_size // self.patch_size) ** 2

        self.patch_dim = patch_size * patch_size
        self.patch_embeddings = nn.Sequential(nn.Conv2d(1, 1, kernel_size=patch_size, stride=patch_size),
                                              nn.Conv2d(1, embed_dim, kernel_size=1, stride=1))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)

        # embedding
        patches = self.patch_embeddings(x)
        patches = patches.flatten(2).transpose(1, 2)

        # CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)

        # attn layers
        x, _ = self.attn1(x, x, x)
        attn_out, attn_weights = self.attn2(x, x, x)

        # attn map
        cls_attention = attn_weights[:, 0, 1:]
        cls_attention = cls_attention.reshape(batch_size, self.image_size // self.patch_size, self.image_size // self.patch_size)
        cls_attention = cls_attention.unsqueeze(1)
        attn_map = F.interpolate(cls_attention, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return attn_map, attn_out


class BottleNeckBlock(nn.Module):
    def __init__(self, image_size, input_channels, output_channels, stride):
        super().__init__()
        self.image_size = image_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.mid_channels = output_channels // 8

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(input_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)

        self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_channels)

        self.conv3 = nn.Conv2d(self.mid_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.relu(out)
        return out


class MainClassifier(nn.Module):
    def __init__(self, image_size, input_channels, output_channels, num_classes):
        super().__init__()
        self.image_size = image_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_classes = num_classes

        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # self.bottleneck1 = BottleNeckBlock(image_size, input_channels, 32, 1)
        self.bottleneck2 = BottleNeckBlock(image_size//2, 32, 32, 1)
        self.bottleneck3 = BottleNeckBlock(image_size//2, 32, 32, 2)
        self.bottleneck4 = BottleNeckBlock(image_size // 4, 32, 32, 1)
        self.bottleneck5 = BottleNeckBlock(image_size // 4, 32, 32, 1)
        self.bottleneck6 = BottleNeckBlock(image_size // 4, 32, 32, 2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.linear(x)

        return logits


class AuxiliaryClassifier(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim):
        super().__init__()
        self.img_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear((self.num_patches + 1) * self.embed_dim, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_flat = self.flatten(x)
        x = self.linear(x_flat)
        x = self.bn(x)
        logits = self.relu(x)
        return logits
