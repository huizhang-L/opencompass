from opencompass.models import OpenAISDK

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        abbr='vllm-serve-qwen2.5-7b-instruct',
        type=OpenAISDK,
        key='EMPTY', # API key
        openai_api_base='http://localhost:37114/v1/', # 服务地址
        path='qwen2.5-7b-instruct', # 请求服务时的 model name
        tokenizer_path='/mnt/data/models/pretrain_models/Qwen2.5-7B-Instruct', # 请求服务时的 tokenizer name 或 path, 为None时使用默认tokenizer gpt-4
        rpm_verbose=True, # 是否打印请求速率
        meta_template=api_meta_template, # 服务请求模板
        query_per_second=8, # 服务请求速率
        max_out_len=4096, # 最大输出长度
        max_seq_len=4096, # 最大输入长度
        temperature=0.01, # 生成温度
        # tok_p=1, # 生成温度
        batch_size=64, # 批处理大小
        retry=3, # 重试次数
    )
]