from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, GaoKaoMATHEvaluator

# ----------------------------- Model Eval Parameters -----------------------------

naive_model_name = 'llama3.3-70b-instruct' # replace with your model name
naive_model_url = ['http://localhost:37114/v1/'] # Multi-apis for accerlation

# ----------------------------- Detailed Config -----------------------------

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt="You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."),
            ],
            round=[
                # dict(role='HUMAN', prompt='{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.'),
                dict(role='HUMAN', prompt='{problem}\nPlease put your final answer within \\boxed{}.'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=20000),
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

math_eval_cfg = dict(
    evaluator=evaluator,
)

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math',
        path='opencompass/math',
        file_name = 'opencompass_test.json',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
