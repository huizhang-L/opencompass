from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwq-32b-preview-vllm',
        path='/mnt/data/models/pretrain_models/QwQ-32B-Preview',
        model_kwargs=dict(tensor_parallel_size=2),
        max_out_len=4096,
        batch_size=16,
        generation_kwargs=dict(temperature=0.7),
        run_cfg=dict(num_gpus=2),
    )
]
