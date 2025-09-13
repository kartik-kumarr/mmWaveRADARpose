import torch
import torch.nn as nn

class mmPose(nn.Module):
    def __init__(self):
        super(mmPose, self).__init__()

        # --- Encoder with residual connections and BatchNorm ---
        def encoder_block():
            return nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.01),
                nn.Dropout2d(p=0.2),

                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.01),
                nn.Dropout2d(p=0.2),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.01),
                nn.Dropout2d(p=0.2),
            )

        self.encoder1 = encoder_block()
        self.encoder2 = encoder_block()

        # --- Concatenation instead of addition ---
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
        )

        # Smaller spatial pooling for reduced MLP size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()

        # --- Efficient MLP with LayerNorm and GELU ---
        self.mlp = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=0.3),

            nn.Linear(128, 102),  # 2 * 17 * 3
        )

    def forward(self, x, y):
        batchSize = x.size(0)

        x = self.encoder1(x)
        y = self.encoder2(y)

        # Concatenate along channel axis
        merged = torch.cat([x, y], dim=1)

        features = self.conv(merged)
        pooled = self.pool(features)
        flat = self.flatten(pooled)
        output = self.mlp(flat)

        return output.view(batchSize, 2, 17, 3)
    

######################### Gaussian keypoint predictions based model ##########################################

class mmPose(nn.Module):
    def __init__(self):
        super(mmPose, self).__init__()
        def encoder_block():
            return nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.01),
                nn.Dropout2d(p=0.2),

                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.01),
                nn.Dropout2d(p=0.2),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.01),
                nn.Dropout2d(p=0.2),
            )

        self.encoder1 = encoder_block()
        self.encoder2 = encoder_block()

        # --- Concatenation instead of addition ---
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
        )

        # Smaller spatial pooling for reduced MLP size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()

        
    def forward(self, x, y):
        batchSize = x.size(0)

        x = self.encoder1(x)
        y = self.encoder2(y)

        # Concatenate along channel axis
        merged = torch.cat([x, y], dim=1)

        features = self.conv(merged)
        pooled = self.pool(features)
        flat = self.flatten(pooled)
        output = self.mlp(flat)

        return output.view(batchSize, 2, 17, 3)