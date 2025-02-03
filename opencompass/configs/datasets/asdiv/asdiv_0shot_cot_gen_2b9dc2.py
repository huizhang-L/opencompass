from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import AsdivDataset, GaoKaoMATHEvaluator, MATHEvaluator, math_postprocess_v2
# ----------------------------- Model Eval Parameters -----------------------------

naive_model_name = 'llama3.3-70b-instruct' # replace with your model name
naive_model_url = ['http://localhost:37114/v1/'] # Multi-apis for accerlation

# ----------------------------- Detailed Config -----------------------------

asdiv_reader_cfg = dict(
    input_columns=['problem'], 
    output_column='Answer'
)


asdiv_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."),
            ],
            round=[
                dict(role='HUMAN', prompt='{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.'),
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=20000)
)

evaluator = dict(
    type=GaoKaoMATHEvaluator,
    model_name=naive_model_name,
    url=naive_model_url,
    language='en',
    with_postprocess=True,
    post_url=naive_model_url,
    post_model_name=naive_model_name,
)

asdiv_eval_cfg = dict(
    evaluator=evaluator,
)

# asdiv_eval_cfg = dict(
#     evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2),
# )

asdiv_datasets = [
    dict(
        abbr='asdiv',
        type=AsdivDataset,
        path='opencompass/asdiv',
        file_name='asdiv.jsonl',
        reader_cfg=asdiv_reader_cfg,
        infer_cfg=asdiv_infer_cfg,
        eval_cfg=asdiv_eval_cfg
    )
]