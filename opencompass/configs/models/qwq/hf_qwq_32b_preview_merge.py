from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwq-32b-preview-hf',
        path='/mnt/data/user/lv_huijie/_MODELS/qwq-qwen2.5-32b-instruct-merge',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
    )
]