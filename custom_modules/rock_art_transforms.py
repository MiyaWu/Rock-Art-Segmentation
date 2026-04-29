from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs


@TRANSFORMS.register_module()
class PackRockArtInputs(PackDetInputs):
    """
    继承官方的 PackDetInputs，复用其强大的 BBox 和 Mask 处理逻辑。
    我们唯一要做的，就是把 'time_info' 和 'loc_info' 加入到 meta_keys 白名单里。
    """

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction',
                            'time_info', 'loc_info','time_mask','loc_mask')):  # <--- 这里加入了我们的自定义字段

        # 调用父类初始化，传入新的 keys
        super().__init__(meta_keys=meta_keys)
