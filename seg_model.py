"""
"Little U-Net" decoder for disturbance segmentation.

CROMA (frozen) already produced the features: one (768, 15, 15) embedding per tile.
This decoder upsamples those features 8x (15 -> 120) into a per-pixel class map.

  input : (B, 768, 15, 15)   CROMA joint embedding
  output: (B, 7, 120, 120)    class logits per pixel (6 types + background)
  B is the batch size.

Decoder on top of the frozen CROMA backbone. We only train the decoder, not the backbone.
"""

import torch.nn as nn


class SegDecoder(nn.Module):
    def __init__(self, in_dim=768, num_classes=7):
        super().__init__() #initialize nn.Module

        #entry cin, output cout)
        def up_block(cin, cout):
            # upsample x2, then a conv to refine
            return nn.Sequential(
                #upsample duplicates each pixel to make it 2x bigger, with interpolation nearest, align corners per their centers
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                #convolution 3x3 with padding 1 to keep the same spatial dimensions
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                # normalize the features after convolution
                nn.BatchNorm2d(cout),
                # apply non-linearity
                nn.ReLU(inplace=True), #inplace=True means it will modify the input tensor directly, saving memory by not creating a new tensor for the output of ReLU
            )

        # 1x1 conv to reduce 768 channels before upsampling
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up1 = up_block(256, 128)   # 15  -> 30
        self.up2 = up_block(128, 64)    # 30  -> 60
        self.up3 = up_block(64, 32)     # 60  -> 120
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)  # per-pixel logits

    def forward(self, x):               # x: (B, 768, 15, 15)
        x = self.proj(x)                # (B, 256, 15, 15)
        x = self.up1(x)                 # (B, 128, 30, 30)
        x = self.up2(x)                 # (B, 64, 60, 60)
        x = self.up3(x)                 # (B, 32, 120, 120)
        return self.head(x)             # (B, 7, 120, 120)


if __name__ == "__main__":
    import torch
    from seg_dataset import NUM_CLASSES
    model = SegDecoder(in_dim=768, num_classes=NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(2, 768, 15, 15)     # fake batch of 2 tiles
    y = model(x)
    print(f"params: {n_params:,}")
    print(f"input : {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
