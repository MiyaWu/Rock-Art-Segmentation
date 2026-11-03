# =========================================================
# 0. 导入自定义模块 (这是最关键的一步)
# =========================================================
custom_imports = dict(
    imports=[
        'custom_modules.rock_art_mask_rcnn',
        'custom_modules.rock_art_transforms',
        'custom_modules.rock_art_dataset'
    ],
    allow_failed_imports=False)
# ================== 自动注入的类别列表 ==================

classes = (
    'Sheep_Goat',       # ID 1
    'Anthropomorph',    # ID 2
    'Symbol',           # ID 3
    'Cattle_Bovine',    # ID 4
    'Deer_Cervid',      # ID 5
    'Horse_Equid',      # ID 6
    'Face_Mask',        # ID 7
    'Rider',            # ID 8
    'Camel',            # ID 9
    'Utensil_Instrument', # ID 10
    'Feline'            # ID 11
)
try:
    classes
except NameError:
    classes = ('rock_art_placeholder',)
# ========================================================

# =========================================================
# 1. 运行环境设置 (Runtime Settings) - 原 default_runtime.py
# =========================================================
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # 每 5 轮保存一次常规权重
        max_keep_ckpts=3,  # 最多保留 3 个常规权重，节省空间
        save_best='coco/bbox_mAP',  # 监控指标：bbox mAP (也可以设为 'coco/segm_mAP')
        rule='greater'  # 规则：越大越好
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
   # visualization=dict(type='DetVisualizationHook')
    visualization=dict(
        type='DetVisualizationHook',
        draw=True,             # 开启画图：必须为 True
        interval=1,            # 每一张图都画 (如果要跳过一些图，可以设为 10 或 50)
        show=False,            # 不要弹窗显示 (服务器上弹窗会报错)
        wait_time=2,           # 没弹窗时这个参数没用
        test_out_dir='work_dirs/vis_results_fixed' # 【指定保存路径】
    )
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer',alpha=0.3)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = 'checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
resume = False

# =========================================================
# 2. 训练计划设置 (Schedule Settings) - 原 schedule_1x.py
# =========================================================
# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
)

# 学习率
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=72,  # 增加到 100 epochs
        by_epoch=True,
        milestones=[50, 60],
        gamma=0.1)
]

# 训练循环
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=72, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

auto_scale_lr = dict(enable=False, base_batch_size=16)

# =========================================================
# 3. 模型结构设置 (Model Settings)
# =========================================================
num_classes = 11

model = dict(
    type='RockArtMaskRCNN',

    fusion_cfg=dict(
        visual_channels=256,
        # 根据实际情况修改 num_time_classes 和 num_loc_classes
        num_time_classes=31,
        num_loc_classes=74,
        embed_dim=64
    ),

    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_mask=True,
        mask_pad_value=0,
    ),

    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        # 引入 DCNv2
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        # 指定在哪些 stage 使用 DCN (这里是在 stage 2, 3, 4 启用)
        stage_with_dcn=(False, True, True, True)
    ),

    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),

    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),

    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5))
)

# =========================================================
# 4. 数据集设置 (Dataset Settings) - 原 coco_instance.py
#    + 你的自定义修改
# =========================================================
dataset_type = 'RockArtDataset'
data_root = 'data/rock_art/'
backend_args = None

# Albumentations 是目前最强的图像增强库，MMDetection 原生支持
albu_train_transforms = [
    # --- 模拟遮挡 (替代 CopyPaste 的部分功能) ---
    # 在图片上随机挖掉一些矩形洞。
    # 这能迫使模型通过局部特征（而非整体轮廓）去识别物体，
    # 效果类似 Cutout/CopyPaste
    dict(
        type='CoarseDropout',
        num_holes_range=(1, 8),
        hole_height_range=(16, 32),
        hole_width_range=(16, 32),
        fill_value=(114, 114, 114),
        p=0.5),
    # --- 像素级增强 (增强纹理鲁棒性) ---
    dict(
        type='RandomBrightnessContrast',  # 随机亮度对比度
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='GaussianBlur',  # 高斯模糊 (模拟对焦不准)
        blur_limit=3,
        p=0.1),
    dict(
        type='GaussNoise',  # 高斯噪声 (模拟岩石表面颗粒)
        var_limit=(10.0, 50.0),
        p=0.1)
]

# 【定义 Pipeline 变量】
image_scale = (1333, 800)

# 4. 构建 Pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # 使用 mask，poly2mask=True 保证输出为位图，配合 Albu 使用最稳
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=True),

    # 基础翻转
    dict(type='RandomFlip', prob=0.5),

    # --- 插入 Albumentations 增强 ---
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels','gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        # Albu 处理完后，Mask 依然是对齐的，不需要修正
        skip_img_without_anno=True),

    # 过滤掉因为旋转而消失的空图
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),

    # 多尺度训练 (Multi-scale Resize)
    dict(
        type='RandomResize',
        #scale=[(1333, 480), (1333, 900)],  # 在这个范围内随机抖动
        scale=[(1333, 600), (1333, 1200)],
        keep_ratio=True,
        clip_object_border=True),

    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackRockArtInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'time_info', 'loc_info','time_mask','loc_mask')
    )
]
# =========================================================
# 6. 数据加载器 (Dataloader)
# =========================================================
# 5. 数据加载器 (回归最简单的配置，没有任何 Wrapper)
train_dataloader = dict(
    batch_size=4,  # 根据你的显存调整
    num_workers=8,
    persistent_workers=True,

    # 依然推荐使用 ClassAwareSampler 解决长尾问题
    sampler=dict(type='ClassAwareSampler'),

    # 这里的 batch_sampler 也可以用默认的 AspectRatioBatchSampler 了
    batch_sampler=dict(type='AspectRatioBatchSampler'),

    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,  # 【修改】你的路径
        ann_file='annotations/train.json',  # 【修改】你的路径
        data_prefix=dict(img='images/'),  # 【修改】你的路径

        # 基础过滤
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        metainfo=dict(classes=classes)
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args,
    classwise=True)
test_evaluator = val_evaluator