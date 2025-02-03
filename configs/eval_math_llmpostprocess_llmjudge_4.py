from mmengine.config import read_base

with read_base():
    # dataset for qwq and merged model
    # from opencompass.configs.datasets.math.math_0shot_notcot_v2_gen_31d777 import math_datasets
    # from opencompass.configs.datasets.asdiv.asdiv_0shot_notcot_gen_2b9dc2 import asdiv_datasets
    # from opencompass.configs.datasets.gsm8k.gsm8k_0shot_notcot_v2_gen_a58960 import gsm8k_datasets
    # from opencompass.configs.datasets.aime90.aime90_0shot_notcot_gen_2b9dc2 import aime90_datasets
    # from opencompass.configs.datasets.gpqa.gpqa_0shot_nocot_gen_772ea0 import gpqa_datasets

    # dataset for instruct model
    # from opencompass.configs.datasets.math.math_0shot_cot_v2_gen_31d777 import math_datasets
    from opencompass.configs.datasets.asdiv.asdiv_0shot_cot_gen_2b9dc2 import asdiv_datasets
    # from opencompass.configs.datasets.gsm8k.gsm8k_0shot_cot_v2_gen_a58960 import gsm8k_datasets
    # from opencompass.configs.datasets.aime90.aime90_0shot_cot_gen_2b9dc2 import aime90_datasets
    # from opencompass.configs.datasets.gpqa.gpqa_0shot_cot_gen_772ea0 import gpqa_datasets

    # 选择一个感兴趣的模型
    # from opencompass.configs.models.openai.vllm_serve_qwq import models as vllm_serve_qwq_model
    # from opencompass.configs.models.openai.vllm_serve_qwq_qwen2_5_merge import models as vllm_serve_qwq_qwen2_5_merge_model
    # from opencompass.configs.models.openai.vllm_serve_qwen2_5_32b_instruct import models as vllm_serve_qwen2_5_32b_instruct_model
    from opencompass.configs.models.openai.vllm_serve_qwen2_5_7b_instruct import models as vllm_serve_qwen2_5_7b_instruct_model


eval_model_name = 'llama3.3-70b-instruct'
postprocessor_model_name = 'llama3.3-70b-instruct'
eval_model_urls = ['http://localhost:37114/v1/']
postprocessor_model_urls = ['http://localhost:37114/v1/']

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])


for dataset in datasets:
    if dataset['abbr'] != 'gsm8k':
        dataset['eval_cfg']['evaluator']['model_name'] = eval_model_name
        dataset['eval_cfg']['evaluator']['url'] = eval_model_urls
        dataset['eval_cfg']['evaluator']['post_url'] = postprocessor_model_urls
        dataset['eval_cfg']['evaluator']['post_model_name'] = postprocessor_model_name


# -------------Inferen Stage ----------------------------------------

from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=LocalRunner,
        max_num_workers=256,
        task=dict(type=OpenICLEvalTask)
    ),
)
work_dir = './outputs/eval_math_final_4'