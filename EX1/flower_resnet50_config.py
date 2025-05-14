_base_ = 'mmpretrain::resnet/resnet50_8xb32_in1k.py'

model = dict(head=dict(num_classes=5))

load_from = 'checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth'

data_root = 'flower_dataset'           

train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        _delete_=True,
        type='CustomDataset',
        data_root=data_root,
        data_prefix='train',           
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=224),
            dict(type='RandomFlip', direction='horizontal', prob=0.5),
            dict(type='PackInputs'),
        ],
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type='default_collate'),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        _delete_=True,
        type='CustomDataset',
        data_root=data_root,
        data_prefix='val',             
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs'),
        ],
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type='default_collate'),
)

test_dataloader = val_dataloader
test_cfg = dict()
test_evaluator = dict(type='Accuracy', topk=(1, 5))

param_scheduler = dict(
    _delete_=True,
    type='CosineAnnealingLR',
    T_max=50,
    by_epoch=True,
)

train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
)

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(
        _scope_='mmpretrain',
        type='AdamW',
        lr=0.001,
        weight_decay=0.05
    )
)
