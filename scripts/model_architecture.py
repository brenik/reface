import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial Attention - focuses on WHERE (face location)"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module - combines channel + spatial"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with BatchNorm and optional CBAM attention"""
    def __init__(self, in_channels, out_channels, stride=1, use_attention=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.cbam = CBAM(out_channels) if use_attention else None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.cbam is not None:
            out = self.cbam(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AgeGenderNet(nn.Module):
    """
    Custom CNN for Age and Gender prediction
    ~50M parameters, ~100MB .pt file
    """
    def __init__(self):
        super(AgeGenderNet, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 1: 64 filters, 2 blocks
        self.stage1 = self._make_stage(64, 64, 2, stride=1, use_attention=False)

        # Stage 2: 128 filters, 2 blocks
        self.stage2 = self._make_stage(64, 128, 2, stride=2, use_attention=False)

        # Stage 3: 256 filters, 3 blocks (start face detection)
        self.stage3 = self._make_stage(128, 256, 3, stride=2, use_attention=True)

        # Stage 4: 384 filters, 2 blocks (focus on face features)
        self.stage4 = self._make_stage(256, 384, 2, stride=2, use_attention=True)

        # Final attention on full feature map
        self.final_attention = CBAM(384)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Shared dense layers
        self.shared = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Age head (regression: 0-100)
        self.age_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # Gender head (binary classification: 0 or 1)
        self.gender_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Emotion head (8 classes: anger, contempt, disgust, fear, happiness, neutral, sadness, surprise)
        self.emotion_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 8),
            nn.Softmax(dim=1)
        )

        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, use_attention=False):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_attention=False))
        for i in range(1, num_blocks):
            # Add attention to last block in stage
            add_attention = use_attention and (i == num_blocks - 1)
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, use_attention=add_attention))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.final_attention(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.shared(x)

        age = self.age_head(x)
        gender = self.gender_head(x)
        emotion = self.emotion_head(x)

        return age, gender, emotion


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = AgeGenderNet()
    model.eval()

    print(f"Total parameters: {count_parameters(model):,}")

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        age_out, gender_out, emotion_out = model(dummy_input)
        print(f"Age output shape: {age_out.shape}")
        print(f"Gender output shape: {gender_out.shape}")
        print(f"Emotion output shape: {emotion_out.shape}")

    model_size_mb = count_parameters(model) * 4 / (1024 ** 2)
    print(f"Estimated model size: {model_size_mb:.2f} MB (FP32)")