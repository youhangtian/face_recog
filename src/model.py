import torch
import torch.nn as nn 
import torch.nn.functional as F

from src.backbones import get_backbone 


class RecogModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.backbone = get_backbone(cfg.backbone)
        self.loss_fn = CircleLoss(
            cfg.gamma,
            cfg.m,
            num_classes,
            cfg.backbone.num_features
        )

        if cfg.ckpt:
            d = torch.load(cfg.ckpt)
            self.load_state_dict(d)

    def forward(self, imgs, labels=None, masks=None):
        features, attn = self.backbone(imgs, masks=masks, return_attn=True)

        if labels is None: 
            return features
        
        loss = self.loss_fn(features, labels)
        return features, attn, loss
    
    @torch.no_grad()
    def save_backbone(self, path):
        torch.save(self.backbone.state_dict(), path)

    @torch.no_grad()
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        

class CircleLoss(nn.Module):
    def __init__(
        self,
        gamma,
        m,
        num_classes,
        num_features
    ):
        super().__init__()
        self.gamma = gamma
        self.m = m
        self.num_classes = num_classes

        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, num_features)))

    def forward(self, features, labels):
        weight_norm = F.normalize(self.weight)
        logits = F.linear(features, weight_norm)
        logits = logits.clamp(-1, 1)

        class_tensor = torch.arange(self.num_classes).long().to(labels.device)
        p_index = class_tensor.unsqueeze(0) == labels.unsqueeze(1)
        s_p = logits[p_index]
 
        # logits[p_index] = s_p - self.m
        # logits = logits * self.gamma 

        # loss = F.cross_entropy(logits, labels)
        # return loss 

        n_index = p_index.logical_not()
        s_n = logits[n_index]

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m 

        logits[p_index] = alpha_p * (s_p - delta_p) * self.gamma 
        logits[n_index] = alpha_n * (s_n - delta_n) * self.gamma 

        logits_diff = logits - logits[p_index].unsqueeze(1)
        loss = torch.logsumexp(logits_diff, dim=1).mean()
        return loss
    
