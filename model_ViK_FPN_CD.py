import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

# --- 1. RBF-KAN Core (Eq. 3, Eq. 4) ---
class RBFLinear(nn.Module):
   
    
    def __init__(self, in_features, out_features, num_grids=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids

        # Learnable grid parameters (Table 2: "Learnable RBF centers and scales")
        grid = torch.linspace(-2.0, 2.0, num_grids)
        init_mu = grid.repeat(in_features).view(in_features, num_grids)
        init_sigma = torch.full((in_features, num_grids), (grid[1] - grid[0]).abs())
        self.mu = nn.Parameter(init_mu.clone())
        self.sigma = nn.Parameter(init_sigma.clone())

        self.spline_weight = nn.Parameter(torch.randn(in_features, num_grids, out_features) * 0.02)

    def forward(self, x):
        # x shape: (B, N, C)
        x_exp = x.unsqueeze(-1)
        # sigma is learnable and appears in a denominator; keep it strictly
        # positive with a small floor for numerical stability (Eq. 3 itself
        # does not constrain the sign of a *fixed* sigma, but a trainable
        # sigma requires this safeguard to avoid division-by-zero/sign-flip).
        safe_sigma = self.sigma.abs() + 1e-6
        basis = torch.exp(-0.5 * ((x_exp - self.mu) / safe_sigma) ** 2)          # Eq. 3
        return torch.einsum('bnfg,fgo->bno', basis, self.spline_weight)          # Eq. 4


class StandardMixer(nn.Module):
    

    def __init__(self, dim, patch_size=4, num_grids=8):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.pw(self.dw(x))


# --- 2. Change-Aware Components ---
class MultiPatchRBFKAN(nn.Module):
  
    def __init__(self, dim, patch_size=4, num_grids=8):
        super().__init__()
        self.ps = patch_size
        self.dim = dim
        self.patch_mixer = RBFLinear(patch_size ** 2, patch_size ** 2, num_grids=num_grids)

    def forward(self, x):
        B, C, H, W = x.shape
        # T = Unfold(E^{(i-1)}) in R^{N x P^2}                                  (Eq. 2)
        out = x.view(B, C, H // self.ps, self.ps, W // self.ps, self.ps) \
               .permute(0, 1, 2, 4, 3, 5).reshape(-1, (H * W) // (self.ps ** 2), self.ps ** 2)
        out = self.patch_mixer(out)                                             # Eq. 3-4
        out = out.view(B, C, H // self.ps, W // self.ps, self.ps, self.ps) \
                 .permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)
        return out


class LayerNorm2d(nn.Module):
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ChangeInteraction(nn.Module):
   

    def __init__(self, dim, use_cim=True):
        super().__init__()
        self.use_cim = use_cim
        if use_cim:
            self.fusion = nn.Sequential(
                nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1),  # Conv_3x3: triple-stream -> dim
                nn.GELU(),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, kernel_size=1),                  # Conv_1x1
            )
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),  # dual-stream (no diff term)
                nn.GELU(),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, kernel_size=1),
            )

    def forward(self, f1, f2):
        if self.use_cim:
            diff = torch.abs(f1 - f2)                                # Eq. 11
            concat = torch.cat([f1, f2, diff], dim=1)                 # Eq. 12
            return self.fusion(concat)                                # Eq. 13
        else:
            concat = torch.cat([f1, f2], dim=1)
            return self.fusion(concat)


# --- 3. Backbone Blocks (ViK-Block, Sec. 3.2.2 Steps A-D) ---
class MPRBFBlock(nn.Module):
   
    def __init__(self, dim, drop_path=0.1, use_rbf_kan=True):
        super().__init__()
        self.mixer = MultiPatchRBFKAN(dim) if use_rbf_kan else StandardMixer(dim)
        self.norm1 = LayerNorm2d(dim)              # Eq. 5: LayerNorm(y_ViK)
        self.proj1 = nn.Conv2d(dim, dim, 1)         # Eq. 5: Proj(...)
        self.norm2 = nn.GroupNorm(1, dim)           # Eq. 6: GroupNorm(X1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        y_vik = self.mixer(x)                                              # Eq. 2-4
        x = x + self.drop_path(self.proj1(self.norm1(y_vik)))              # Eq. 5
        x = x + self.drop_path(self.mlp(self.norm2(x)))                    # Eq. 6
        return x


# --- 4. The Full Network ---

class ViK_CD(nn.Module):

    def __init__(self, num_classes=2, dims=[64, 128, 256, 512],
                 use_rbf_kan=True, use_cim=True, use_fpn_decoder=True):
        super().__init__()
        self.use_fpn_decoder = use_fpn_decoder

        # Initial embedding: overlapping 7x7 conv, stride 2                     (Eq. 1)
        self.patch_embed = nn.Conv2d(3, dims[0], kernel_size=7, stride=2, padding=3)

       
        self.stage1 = nn.Sequential(*[MPRBFBlock(dims[0], use_rbf_kan=use_rbf_kan) for _ in range(2)])
        self.down1 = nn.Conv2d(dims[0], dims[1], 3, stride=2, padding=1)

        self.stage2 = nn.Sequential(*[MPRBFBlock(dims[1], use_rbf_kan=use_rbf_kan) for _ in range(2)])
        self.down2 = nn.Conv2d(dims[1], dims[2], 3, stride=2, padding=1)

        self.stage3 = nn.Sequential(*[MPRBFBlock(dims[2], use_rbf_kan=use_rbf_kan) for _ in range(4)])
        self.down3 = nn.Conv2d(dims[2], dims[3], 3, stride=2, padding=1)

        self.stage4 = nn.Sequential(*[MPRBFBlock(dims[3], use_rbf_kan=use_rbf_kan) for _ in range(2)])
      

        
        self.change_interactors = nn.ModuleList([ChangeInteraction(d, use_cim=use_cim) for d in dims])

  
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lateral = nn.ModuleList([nn.Conv2d(d, 128, 1) for d in dims])   # Eq. 14
        if not use_fpn_decoder:
          
            self.no_fpn_proj = nn.Conv2d(dims[3], 128, 1)

        
        self.final_refine = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward_encoder(self, x):
        f1 = self.stage1(self.patch_embed(x))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        f4 = self.stage4(self.down3(f3))
        return [f1, f2, f3, f4]

    def forward(self, img1, img2):
       
        feats1 = self.forward_encoder(img1)
        feats2 = self.forward_encoder(img2)

   
        c_feats = [self.change_interactors[i](feats1[i], feats2[i]) for i in range(4)]

        if self.use_fpn_decoder:
            
            p4 = self.lateral[3](c_feats[3])                        
            p3 = self.lateral[2](c_feats[2]) + self.up(p4)          
            p2 = self.lateral[1](c_feats[1]) + self.up(p3)         
            p1 = self.lateral[0](c_feats[0]) + self.up(p2)          
        else:
            
            p1 = F.interpolate(self.no_fpn_proj(c_feats[3]), scale_factor=8,
                                mode='bilinear', align_corners=True)

      
        out = self.final_refine(p1)
        return F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)


