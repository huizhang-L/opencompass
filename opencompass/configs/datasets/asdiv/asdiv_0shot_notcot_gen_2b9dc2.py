from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import Aime90Dataset, GaoKaoMATHEvaluator, math_postprocess_v2
# ----------------------------- Model Eval Parameters -----------------------------

naive_model_name = 'llama3.3-70b-instruct' # replace with your model name
naive_model_url = ['http://localhost:37114/v1/'] # Multi-apis for accerlation

# ----------------------------- Detailed Config -----------------------------

aime90_reader_cfg = dict(
    input_columns=['problem'], 
    output_column='expected_answer'
)


aime90_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt="You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."),
            ],
            round=[
                dict(role='HUMAN', prompt='{problem}\nRemember to put your final answer within \\boxed{}.'),
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

aime90_eval_cfg = dict(
    evaluator=evaluator,
)

aime90_datasets = [
    dict(
        abbr='aime90',
        type=Aime90Dataset,
        path='opencompass/aime90',
        file_name='aime.jsonl',
        reader_cfg=aime90_reader_cfg,
        infer_cfg=aime90_infer_cfg,
        eval_cfg=aime90_eval_cfg
    )
]