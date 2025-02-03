from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, MATHEvaluator
from opencompass.datasets import GaoKaoMATHEvaluator, math_postprocess_v2
# ----------------------------- Model Eval Parameters -----------------------------

naive_model_name = 'llama3.3-70b-instruct' # replace with your model name
naive_model_url = ['http://localhost:37114/v1/'] # Multi-apis for accerlation

# ----------------------------- Detailed Config -----------------------------
gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt="You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."),
            ],
            round=[
                dict(role='HUMAN', prompt='{question}\nPlease put your final answer within \\boxed{}.'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=20000),
)

# evaluator = dict(
#     type=GaoKaoMATHEvaluator,
#     model_name=naive_model_name,
#     url=naive_model_url,
#     language='en',
#     with_postprocess=True,
#     post_url=naive_model_url,
#     post_model_name=naive_model_name,
# )

# gsm8k_eval_cfg = dict(
#     evaluator=evaluator,
#     # pred_postprocessor=dict(type=math_postprocess_v2),
#     dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
# )

gsm8k_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'),
    pred_postprocessor=dict(type=math_postprocess_v2),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)

gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
