"""
ROI classifier: pretrained nnU-Net backbone -> region-masked pooling ->
location-aware transformer -> per-location, presence and type heads.

RSNA -> TopAneu changes are marked [TOPANEU]:
  * n_loc is 52 rather than 13, so the location-token embedding table and the
    transformer sequence length grow accordingly;
  * locations and vessel regions are decoupled -- several of the 52 locations
    may live on one vessel segment, so pooling runs once per vessel class and
    tokens gather from that table;
  * an optional aneurysm-type head, since TopAneu ships type_masks and RSNA had
    no equivalent label.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from topaneu_rsna.cls.backbone import NnUNetTruncatedBackbone
from topaneu_rsna.cls.pooling import VesselRegionMaskedPooling


class AneurysmRoiNet(nn.Module):
    def __init__(self, spec, cfg, nnunet_model_dir, seg_fold="all", pretrained=None):
        super().__init__()
        self.cfg = cfg
        self.n_loc = spec.n_loc
        self.n_vessel = spec.n_vessel
        self.n_type = spec.n_type
        self.register_buffer(
            "vessel_index",
            torch.as_tensor(spec.vessel_label_for_location(), dtype=torch.long) - 1,
            persistent=True)

        self.backbone = NnUNetTruncatedBackbone(
            nnunet_model_dir=nnunet_model_dir, fold=seg_fold,
            pretrained=cfg.pretrained if pretrained is None else pretrained,
            num_truncate_stages=cfg.num_truncate_stages,
            sphere_mid_channels=cfg.sphere_mid_channels)

        dec_c = self.backbone.out_channels
        enc_c = self.backbone.encoder_channels

        self.pool_loc = VesselRegionMaskedPooling(
            dec_c, enc_c, cfg.proj_mask_dim, cfg.proj_global_dim, cfg.branch_norm)
        self.pool_ap = VesselRegionMaskedPooling(
            dec_c, enc_c, cfg.proj_mask_dim, cfg.proj_global_dim, cfg.branch_norm)

        n_branch = 1 + int(cfg.num_extra_mask_branches)
        tok_dim = n_branch * cfg.proj_mask_dim + cfg.proj_global_dim
        d = cfg.transformer_embed_dim

        self.loc_proj = nn.Linear(tok_dim, d)
        self.loc_token_embed = nn.Parameter(torch.zeros(self.n_loc, d))
        nn.init.trunc_normal_(self.loc_token_embed, std=0.02)
        self.loc_dropout = nn.Dropout(cfg.transformer_dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.transformer_heads, dim_feedforward=d * 2,
            dropout=cfg.transformer_dropout, batch_first=True, norm_first=True)
        self.loc_transformer = nn.TransformerEncoder(layer, cfg.transformer_layers)
        self.loc_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

        self.ap_head = nn.Sequential(
            nn.LayerNorm(tok_dim), nn.Linear(tok_dim, d),
            nn.GELU(), nn.Dropout(cfg.transformer_dropout), nn.Linear(d, 1))

        self.type_head = (nn.Sequential(
            nn.LayerNorm(tok_dim), nn.Linear(tok_dim, d),
            nn.GELU(), nn.Dropout(cfg.transformer_dropout), nn.Linear(d, self.n_type))
            if (cfg.use_type_head and self.n_type > 0) else None)

    def forward(self, image, vessel_m2, vessel_m3=None):
        out = self.backbone(image)
        dec, enc = out["dec_feat"], out["enc_feat"]

        v2, valid2 = self.pool_loc.pool_regions(dec, vessel_m2, self.n_vessel)
        branches = [self.pool_loc.project_mask(v2)[:, self.vessel_index, :]]
        valid = valid2[:, self.vessel_index]

        if self.cfg.num_extra_mask_branches > 0:
            src = vessel_m3 if vessel_m3 is not None else vessel_m2
            v3, _ = self.pool_loc.pool_regions(dec, src, self.n_vessel)
            branches.append(self.pool_loc.project_mask(v3)[:, self.vessel_index, :])

        g = self.pool_loc.project_global(enc, self.n_loc)
        tokens = torch.cat(branches + [g], dim=-1).to(dec.dtype)

        x = self.loc_proj(tokens) + self.loc_token_embed.unsqueeze(0)
        x = self.loc_transformer(self.loc_dropout(x))
        loc_logits = self.loc_head(x).squeeze(-1)

        # presence: pool over the union of every vessel region
        u2 = self.pool_ap.project_mask(self.pool_ap.pool_union(dec, vessel_m2)[:, None])
        u_branches = [u2]
        if self.cfg.num_extra_mask_branches > 0:
            src = vessel_m3 if vessel_m3 is not None else vessel_m2
            u_branches.append(
                self.pool_ap.project_mask(self.pool_ap.pool_union(dec, src)[:, None]))
        gu = self.pool_ap.project_global(enc, 1)
        ap_tok = torch.cat(u_branches + [gu], dim=-1).to(dec.dtype).squeeze(1)

        res = {
            "loc_logits": loc_logits,
            "ap_logit": self.ap_head(ap_tok).squeeze(-1),
            "logits_sphere": out["logits_sphere"],
            "token_valid": valid,
            "type_logits": self.type_head(ap_tok) if self.type_head is not None else None,
        }
        return res


def build_model(spec, cfg, nnunet_model_dir, seg_fold="all", pretrained=None):
    return AneurysmRoiNet(spec, cfg, nnunet_model_dir, seg_fold, pretrained)
