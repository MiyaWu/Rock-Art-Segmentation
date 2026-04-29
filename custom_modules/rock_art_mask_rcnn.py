import torch
import torch.nn as nn
import numpy as np
from mmdet.registry import MODELS
from mmdet.models.detectors import MaskRCNN


class TimeLocationFusion(nn.Module):
    def __init__(self,
                 visual_channels=256,
                 num_time_classes=50,  # 时间有多少个类别
                 num_loc_classes=50,  # 地点有多少个类别
                 embed_dim=64):  # 嵌入向量的长度
        super().__init__()

        # 1. 定义嵌入层 (Input: ID -> Output: Embed_Dim)
        # Padding_idx=0 表示如果输入ID为0，则输出全0向量（可选，但在我们的Mask方案中不是必须的）
        self.time_emb = nn.Embedding(num_time_classes, embed_dim)
        self.loc_emb = nn.Embedding(num_loc_classes, embed_dim)

        # 2. 投影层 (Input: Embed_Dim -> Output: 256)
        # 将嵌入向量映射到与 FPN 通道数一致
        self.time_project = nn.Sequential(
            nn.Linear(embed_dim, visual_channels),
            nn.LayerNorm(visual_channels),
            nn.ReLU()
        )

        self.loc_project = nn.Sequential(
            nn.Linear(embed_dim, visual_channels),
            nn.LayerNorm(visual_channels),
            nn.ReLU()
        )

        # 3. 融合后的激活函数
        self.fusion_act = nn.ReLU()

    def forward(self, visual_feats, time_ids, loc_ids, time_mask, loc_mask):
        """
        visual_feats: List[Tensor], FPN特征
        time_ids: [B], LongTensor, 时间类别索引
        loc_ids:  [B], LongTensor, 地点类别索引
        time_mask: [B, 1], FloatTensor, 0表示未知，1表示已知
        loc_mask:  [B, 1], FloatTensor, 0表示未知，1表示已知
        """

        # 1. 获取嵌入向量 [B, embed_dim]
        t_feat = self.time_emb(time_ids)
        l_feat = self.loc_emb(loc_ids)

        # 2. 投影到视觉维度 [B, 256]
        t_feat = self.time_project(t_feat)
        l_feat = self.loc_project(l_feat)

        # 3. 应用 Mask (核心步骤)
        # 如果 mask 为 0，则该特征变为全 0，不产生干扰
        t_feat = t_feat * time_mask
        l_feat = l_feat * loc_mask

        # 4. 得到全局上下文 (Global Context)
        # 此时包含：已知的时间特征 + 已知的地点特征
        global_context = t_feat + l_feat  # [B, 256]

        # 5. 注入到图像特征中
        new_feats = []
        for feat in visual_feats:
            # feat: [B, 256, H, W]
            # context: [B, 256, 1, 1] (广播)
            fused_feat = feat + global_context.unsqueeze(-1).unsqueeze(-1)
            new_feats.append(self.fusion_act(fused_feat))

        return tuple(new_feats)


@MODELS.register_module()
class RockArtMaskRCNN(MaskRCNN):
    def __init__(self,
                 fusion_cfg=dict(
                     visual_channels=256,
                     num_time_classes=31,
                     num_loc_classes=74,
                     embed_dim=64
                 ),
                 **kwargs):
        super().__init__(**kwargs)
        self.fusion_module = TimeLocationFusion(**fusion_cfg)

    def extract_feat(self, batch_inputs: torch.Tensor, batch_data_samples=None):
        # 1. 提取基础图像特征
        x = super().extract_feat(batch_inputs)

        if batch_data_samples is None:
            return x

        device = batch_inputs.device

        try:
            # 2. 提取数据
            # 注意：ID 必须是整数 (Long/Int)，Mask 必须是浮点数 (Float)

            # --- 提取 ID ---
            time_ids_list = [s.metainfo.get('time_info', [0])[0] for s in batch_data_samples]
            loc_ids_list = [s.metainfo.get('loc_info', [0])[0] for s in batch_data_samples]

            # --- 提取 Mask ---
            # 如果数据里没有 mask 字段，默认设为 0 (未知)，防止报错
            time_mask_list = [s.metainfo.get('time_mask', [0])[0] for s in batch_data_samples]
            loc_mask_list = [s.metainfo.get('loc_mask', [0])[0] for s in batch_data_samples]

            # 3. 转换为 Tensor
            # ID 需要是 Long 类型 (int64) 以被 Embedding 层接受
            time_ids = torch.tensor(time_ids_list, dtype=torch.long, device=device)
            loc_ids = torch.tensor(loc_ids_list, dtype=torch.long, device=device)

            # Mask 需要是 Float 类型以便进行乘法运算
            time_mask = torch.tensor(time_mask_list, dtype=torch.float32, device=device)
            loc_mask = torch.tensor(loc_mask_list, dtype=torch.float32, device=device)

            # 4. 调整维度
            # Mask 需要变成 [Batch, 1] 才能和 [Batch, 256] 广播相乘
            if time_mask.dim() == 1: time_mask = time_mask.unsqueeze(1)
            if loc_mask.dim() == 1: loc_mask = loc_mask.unsqueeze(1)

            # 5. 融合
            x_fused = self.fusion_module(x, time_ids, loc_ids, time_mask, loc_mask)
            return x_fused

        except Exception as e:
            return x