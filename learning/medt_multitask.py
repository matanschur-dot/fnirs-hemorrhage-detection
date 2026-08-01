from __future__ import annotations

import torch
from torch import nn


class GatedAxialAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.source_norm = nn.LayerNorm(channels)
        self.detector_norm = nn.LayerNorm(channels)

        self.source_attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.detector_attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )

        self.source_gate = nn.Parameter(torch.tensor(-1.0))
        self.detector_gate = nn.Parameter(torch.tensor(-1.0))

        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, s, d = x.shape

        source_tokens = x.permute(0, 3, 2, 1).reshape(b * d, s, c)
        normalized = self.source_norm(source_tokens)
        attended, _ = self.source_attention(
            normalized, normalized, normalized, need_weights=False
        )
        source_tokens = source_tokens + torch.sigmoid(self.source_gate) * attended
        x = source_tokens.reshape(b, d, s, c).permute(0, 3, 2, 1)

        detector_tokens = x.permute(0, 2, 3, 1).reshape(b * s, d, c)
        normalized = self.detector_norm(detector_tokens)
        attended, _ = self.detector_attention(
            normalized, normalized, normalized, need_weights=False
        )
        detector_tokens = detector_tokens + torch.sigmoid(self.detector_gate) * attended
        detector_tokens = detector_tokens + self.ffn(self.ffn_norm(detector_tokens))

        return detector_tokens.reshape(b, s, d, c).permute(0, 3, 1, 2)


class LocalGlobalMedTEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int = 96,
        num_heads: int = 8,
        num_blocks: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.axial_blocks = nn.ModuleList(
            [
                GatedAxialAttention(
                    channels=channels,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        local = self.local_branch(stem)

        global_features = stem
        for block in self.axial_blocks:
            global_features = block(global_features)

        joined = torch.cat([local, global_features], dim=1)
        gate = self.fusion_gate(joined)
        fused = self.fusion(joined)
        return gate * fused + (1.0 - gate) * stem


class DualBranchMedT(nn.Module):
    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        voxel_size_mm: float,
        channels: int = 96,
        num_heads: int = 8,
        num_blocks: int = 4,
        dropout: float = 0.10,
        mask_temperature: float = 10.0,
    ) -> None:
        super().__init__()

        self.volume_shape = tuple(int(v) for v in volume_shape)
        self.voxel_size_mm = float(voxel_size_mm)
        self.mask_temperature = float(mask_temperature)

        self.detection_encoder = LocalGlobalMedTEncoder(
            in_channels=4,
            channels=channels,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.localization_encoder = LocalGlobalMedTEncoder(
            in_channels=2,
            channels=channels,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout,
        )

        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        self.detection_head = nn.Sequential(
            nn.LayerNorm(channels + 12),
            nn.Linear(channels + 12, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, 1),
        )

        self.localization_shared = nn.Sequential(
            nn.LayerNorm(channels + 12),
            nn.Linear(channels + 12, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.GELU(),
        )
        self.center_head = nn.Linear(128, 3)
        self.depth_head = nn.Linear(128, 1)
        self.radii_head = nn.Linear(128, 3)

        grids = torch.meshgrid(
            *[torch.arange(size, dtype=torch.float32) for size in self.volume_shape],
            indexing="ij",
        )
        self.register_buffer("grid0", grids[0], persistent=False)
        self.register_buffer("grid1", grids[1], persistent=False)
        self.register_buffer("grid2", grids[2], persistent=False)

    def freeze_detection_branch(self) -> None:
        for module in (self.detection_encoder, self.detection_head):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def unfreeze_detection_branch(self) -> None:
        for module in (self.detection_encoder, self.detection_head):
            for parameter in module.parameters():
                parameter.requires_grad = True

    @staticmethod
    def _absolute_statistics(x: torch.Tensor) -> torch.Tensor:
        baseline = x[:, 0:2]
        current = x[:, 2:4]
        difference = current - baseline

        mean = difference.mean(dim=(2, 3))
        mean_abs = difference.abs().mean(dim=(2, 3))
        rms = torch.sqrt(difference.square().mean(dim=(2, 3)) + 1e-8)
        max_abs = difference.abs().amax(dim=(2, 3))
        positive_fraction = (difference > 0).float().mean(dim=(2, 3))
        cosine = torch.nn.functional.cosine_similarity(
            baseline.flatten(2), current.flatten(2), dim=2
        )
        return torch.cat(
            [mean, mean_abs, rms, max_abs, positive_fraction, cosine],
            dim=1,
        )

    @staticmethod
    def _residual_statistics(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3))
        mean_abs = x.abs().mean(dim=(2, 3))
        rms = torch.sqrt(x.square().mean(dim=(2, 3)) + 1e-8)
        max_abs = x.abs().amax(dim=(2, 3))
        minimum = x.amin(dim=(2, 3))
        maximum = x.amax(dim=(2, 3))
        return torch.cat([mean, mean_abs, rms, max_abs, minimum, maximum], dim=1)

    def _soft_ellipsoid(
        self,
        center: torch.Tensor,
        radii: torch.Tensor,
    ) -> torch.Tensor:
        c0, c1, c2 = [
            center[:, axis].view(-1, 1, 1, 1) for axis in range(3)
        ]
        r0, r1, r2 = [
            radii[:, axis].view(-1, 1, 1, 1).clamp_min(0.75)
            for axis in range(3)
        ]

        distance = (
            ((self.grid0 - c0) / r0).square()
            + ((self.grid1 - c1) / r1).square()
            + ((self.grid2 - c2) / r2).square()
        )
        return torch.sigmoid(self.mask_temperature * (1.0 - distance))

    def forward(
        self,
        x_absolute: torch.Tensor,
        x_residual: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        detection_map = self.detection_encoder(x_absolute)
        detection_features = torch.cat(
            [
                self.pool(detection_map),
                self._absolute_statistics(x_absolute),
            ],
            dim=1,
        )
        class_logit = self.detection_head(detection_features).squeeze(-1)

        localization_map = self.localization_encoder(x_residual)
        localization_features = self.localization_shared(
            torch.cat(
                [
                    self.pool(localization_map),
                    self._residual_statistics(x_residual),
                ],
                dim=1,
            )
        )

        shape = torch.tensor(
            self.volume_shape,
            dtype=x_absolute.dtype,
            device=x_absolute.device,
        )
        center_vox = torch.sigmoid(self.center_head(localization_features)) * (shape - 1.0)
        radii_vox = 0.75 + torch.nn.functional.softplus(
            self.radii_head(localization_features)
        )
        depth_mm = torch.nn.functional.softplus(
            self.depth_head(localization_features).squeeze(-1)
        )

        radii_mm = radii_vox * self.voxel_size_mm
        volume_mm3 = (
            (4.0 / 3.0)
            * torch.pi
            * radii_mm[:, 0]
            * radii_mm[:, 1]
            * radii_mm[:, 2]
        )
        soft_mask = self._soft_ellipsoid(center_vox, radii_vox)

        return {
            "class_logit": class_logit,
            "center_vox": center_vox,
            "depth_mm": depth_mm,
            "radii_vox": radii_vox,
            "volume_mm3": volume_mm3,
            "soft_mask": soft_mask,
        }
