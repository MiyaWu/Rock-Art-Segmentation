from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset
import copy


@DATASETS.register_module()
class RockArtDataset(CocoDataset):
    """
    自定义的岩画数据集类，继承自 CocoDataset。
    功能：
    1. 继承标准 COCO 格式解析能力。
    2. 额外解析 time_info, loc_info 及其对应的 mask。
    """

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """
        重写解析函数。
        raw_data_info: 从 JSON 的 'images' 列表里读取的原始字典
        """
        # 1. 先让父类处理标准的 bbox, mask, file_name 等逻辑
        data_info = super().parse_data_info(raw_data_info)

        # 2. 手动提取自定义字段
        # 使用 .get(key, default) 方法更安全
        # 如果 JSON 里没有这些字段，默认给 [0]，mask 默认给 [0] (表示未知)

        # --- 时间信息 ---
        # time_info: 整数类别 ID
        data_info['time_info'] = raw_data_info.get('time_info', [0])
        # time_mask: 0 (未知) 或 1 (已知)
        data_info['time_mask'] = raw_data_info.get('time_mask', [0])

        # --- 地点信息 ---
        # loc_info: 整数类别 ID
        data_info['loc_info'] = raw_data_info.get('loc_info', [0])
        # loc_mask: 0 (未知) 或 1 (已知)
        data_info['loc_mask'] = raw_data_info.get('loc_mask', [0])

        return data_info