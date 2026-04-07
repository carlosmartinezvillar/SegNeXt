# 1. Inherit
_base_ = [
    '../segnext/large/segnext.large.512x512.ade.160k.py',  # architecture
    '../../configs/_base_/datasets/loveda.py',            # iSAID
    '../_base_/default_runtime.py',                       # Logging/Checkpoints
    '../_base_/schedules/schedule_160k.py'                # 160k iter schedule
]

# 2. Update Model for loveDA (6 classes + background = 7)
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/segnext_base_512x512_ade_160k.pth',prefix='backbone.')),    
    decode_head=dict(
        num_classes=7,
        # average_non_ignore helps small classes
        loss_decode=dict(type='CrossEntropyLoss', avg_non_ignore=True)
    )
)

# 3. Update Dataset Paths and Batch Size
data = dict(
    samples_per_gpu=8,  # Batch size
    workers_per_gpu=4,  # Parallel CPU threads
    train=dict(data_root='data/loveDA'),
    val=dict(data_root='data/loveDA'),
    test=dict(data_root='data/loveDA')
)

# 4. Apply the Linear Scaling Rule for Learning Rate
# # Original SegNeXt is 0.00006 for Batch 16 (8gpu). For Batch 4 (1gpu), we use 0.000015
# optimizer = dict(lr=0.000015)
# Original SegNeXt is 0.00006 for Batch 16 (8gpu). For Batch 8 (2gpu), we use 0.00003
optimizer = dict(lr=0.00003)

# 5. Stabilize Gradients for fp16
optimizer_config = dict(grad_clip=dict(max_norm=0.35, norm_type=2))
fp16 = dict(loss_scale='dynamic')