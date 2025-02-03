from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwq-32b-preview-hf',
        path='/mnt/data/models/pretrain_models/QwQ-32B-Preview',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
    )
]