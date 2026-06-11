"""
Linear-probe segmentation head (CROMA paper protocol: Linear(E_O)).

The simplest possible head on frozen CROMA features: a SINGLE linear layer
applied to each patch token, then bilinear upsampling to the pixel grid.
No learned decoder, almost no parameters.

  input : (B, 768, 15, 15)   CROMA joint embedding (frozen patch tokens)
  output: (B, 7, 120, 120)    class logits per pixel

Same input/output shapes as SegDecoder in seg_model.py, so it is a drop-in
replacement for comparing "tiny linear probe" vs "little U-Net decoder".

A 1x1 conv == a linear layer applied independently to every token, which is
exactly what "Linear(E_O)" means in the paper.
"""

import torch.nn as nn
import torch.nn.functional as F


class LinearProbeSeg(nn.Module):
    def __init__(self, in_dim=768, num_classes=7, out_size=120):
        super().__init__()
        # one linear layer over the 768-dim token -> num_classes (per token)
        self.classifier = nn.Conv2d(in_dim, num_classes, kernel_size=1)
        self.out_size = out_size

    def forward(self, x):                       # x: (B, 768, 15, 15)
        logits = self.classifier(x)             # (B, 7, 15, 15)  <- the linear probe
        logits = F.interpolate(                 # upsample scores to pixel grid
            logits, size=(self.out_size, self.out_size),
            mode="bilinear", align_corners=False,
        )
        return logits                           # (B, 7, 120, 120)


if __name__ == "__main__":
    import torch
    from seg_dataset import NUM_CLASSES
    model = LinearProbeSeg(in_dim=768, num_classes=NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(2, 768, 15, 15)
    y = model(x)
    print(f"params entrenables (linear probe): {n_params:,}")
    print(f"input : {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}   (debe ser (2, {NUM_CLASSES}, 120, 120))")
