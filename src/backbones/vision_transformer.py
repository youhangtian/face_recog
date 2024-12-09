import math
import torch 
import torch.nn as nn 


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False):
        super().__init__()
        self.dim = dim 
        self.num_heads = num_heads
        self.head_dim = dim // num_heads 
        assert self.head_dim * num_heads == dim, f'shape error: {dim}, {num_heads}'

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.o_proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x):
        B, N, D = x.shape 
        assert D == self.dim, f'shape error: {x.shape}, {self.dim}' 
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(2, 3)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)

        output = (attn @ v).transpose(1, 2).reshape(B, N, D)
        output = self.o_proj(output)
        return output, attn


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features 
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x 

 
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob 

    def forward(self, x):
        if self.drop_prob == 0. or not self.training: 
            return x 
        
        keep_prob = 1 - self.drop_prob 
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        output = x.div(keep_prob) * random_tensor.floor_()
        return output 
    

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer)
        
    def forward(self, x):
        x_, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn
    

class PatchEmbed(nn.Module):
    def __init__(self,
                 input_size=(224, 224),
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        self.input_size = input_size 
        self.patch_size = patch_size  
        self.num_patches = (input_size[0] // patch_size) * (input_size[1] // patch_size) 

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x 
    

class VisionTransformer(nn.Module):
    def __init__(self,
                 input_size=(224, 224),
                 patch_size=16,
                 freeze_patch_embed=False,
                 in_chans=3,
                 num_features=512,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.freeze_patch_embed = freeze_patch_embed
        self.num_features = num_features 
        self.embed_dim = embed_dim 

        self.patch_embed = PatchEmbed(
            input_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches 
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=dpr[i],
            norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.features = nn.Linear(embed_dim, num_features, bias=False)
        self.f_norm = norm_layer(num_features)

        nn.init.normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(self._init_weights)

        val = math.sqrt(6. / float(3 * self.patch_embed.patch_size ** 2 + self.embed_dim))
        nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
        nn.init.zeros_(self.patch_embed.proj.bias)

        if self.freeze_patch_embed:
            self.mask_token.requires_grad = False 
            self.patch_embed.proj.weight.requires_grad = False 
            self.patch_embed.proj.bias.requires_grad = False 

    @torch.no_grad()
    def _trunc_normal(self, tensor, mean=0., std=1., a=-2., b=2.):
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2. 

        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self._trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def prepare_tokens(self, x, masks=None):
        B = x.shape[0]

        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x) 
        if self.freeze_patch_embed: x = x.detach()

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        return x
    
    def forward(self, x, masks=None, return_attn=False):
        x = self.prepare_tokens(x, masks)
        for blk in self.blocks:
            x, attn = blk(x)
        x = self.norm(x)

        features = self.f_norm(self.features(x[:, 0]))
        features_norm = nn.functional.normalize(features, dim=-1, p=2)

        if not return_attn: 
            return features 
        
        attn = attn[:, :, 0, 1:]
        return features_norm, attn
