# 1. Inherit
_base_ = [
    '../segnext/base/segnext.base.512x512.ade.160k.py', # architecture
    '../_base_/datasets/5_potsdam.py',                   # 5 potsdam 
    '../_base_/default_runtime.py',                      # Logging/Checkpoints
    '../_base_/schedules/schedule_160k.py'               # 160k iter schedule
]

# 2. Update Model for iSAID (15 classes + background = 16)
model = dict(
    decode_head=dict(
        num_classes=5,
        # average_non_ignore helps small classes
        loss_decode=dict(type='CrossEntropyLoss', avg_non_ignore=True)
    )
)

# 3. Update Dataset Paths and Batch Size
# Adjusted for a single GPU
data = dict(
    samples_per_gpu=4,  # Batch size
    workers_per_gpu=4,  # Parallel CPU threads
    train=dict(data_root='data/5potsdam'),
    val=dict(data_root='data/5potsdam'),
    test=dict(data_root='data/5potsdam')
)

# 4. Apply the Linear Scaling Rule for Learning Rate
# Original SegNeXt is 0.00006 for Batch 16 (8gpu). For Batch 4 (1gpu), use 0.000015
optimizer = dict(lr=0.000015)

# 5. Stabilize Gradients (Crucial for remote sensing)
optimizer_config = dict(grad_clip=dict(max_norm=0.35, norm_type=2))

# python tools/misc/print_config.py configs/my_experiments/segnext_l_isaid_single_gpu.py
# python tools/train.py configs/my_experiments/segnext_l_isaid_single_gpu.py --launcher none