# 1. Inherit
_base_ = [
    '../segnext/base/segnext.base.512x512.ade.160k.py', # architecture
    '../../configs/_base_/datasets/potsdam.py',         # 6 potsdam 
    '../_base_/default_runtime.py',                     # Logging/Checkpoints
    '../_base_/schedules/schedule_160k_adamw.py'              # 160k iter schedule
]

# 2. Update Model for Potsdam
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/segnext_base_512x512_ade_160k.pth',prefix='backbone.')
    ),
    decode_head=dict(
        num_classes=6,
        # average_non_ignore helps small classes
        loss_decode=dict(type='CrossEntropyLoss', avg_non_ignore=True)
    ),
    auxiliary_head=dict(
        num_classes=6  # CRITICAL: If you miss this, it will crash
    )        
)

data = dict(_delete_=True)
dataset_type = 'PotsdamDataset'
data_root = 'data/6potsdam'

# 3. Update Dataset Paths and Batch Size
data = dict(
    samples_per_gpu=8,  # Batch size
    workers_per_gpu=4,  # Parallel CPU threads
    train=dict(data_root='data/6potsdam'),
    val=dict(data_root='data/6potsdam'),
    test=dict(data_root='data/6potsdam')
)

# train_pipeline = None
# test_pipeline = None

# 4. Apply the Linear Scaling Rule for Learning Rate
# # Original SegNeXt is 0.00006 for Batch 16 (8gpu). For Batch 4 (1gpu), use 0.000015
# optimizer = dict(lr=0.000015)
# Original SegNeXt is 0.00006 for Batch 16 (8gpu). For Batch 8 (2gpu), use 0.00003
# optimizer = dict(lr=0.00003)

# 5. Stabilize Gradients for fp16
optimizer_config = dict(grad_clip=dict(max_norm=0.35, norm_type=2))
fp16 = dict(loss_scale='dynamic')

# TEST RUNS WITH:
# python tools/misc/print_config.py configs/my_experiments/segnext_l_isaid_single_gpu.py
# python tools/train.py configs/my_experiments/segnext_l_isaid_single_gpu.py --launcher none