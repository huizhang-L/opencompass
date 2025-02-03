from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GPQADataset, GaoKaoMATHEvaluator, GPQAEvaluator, GPQA_Simple_Eval_postprocess, math_postprocess_v2
# ----------------------------- Model Eval Parameters -----------------------------

naive_model_name = 'llama3.3-70b-instruct' # replace with your model name
naive_model_url = ['http://localhost:37114/v1/'] # Multi-apis for accerlation

# ----------------------------- Detailed Config -----------------------------
# openai_simple_eval prompt
align_prompt = """
Answer the following multiple choice question. Remember to put your final answer within \\boxed{}. The answer only needs to include the option where is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

gpqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."),
            ],
            round=[
                dict(role='HUMAN', prompt=align_prompt),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=20000))

evaluator = dict(
    type=GaoKaoMATHEvaluator,
    model_name=naive_model_name,
    url=naive_model_url,
    language='en',
    with_postprocess=True,
    post_url=naive_model_url,
    post_model_name=naive_model_name,
)
gpqa_eval_cfg = dict(evaluator=evaluator)

# gpqa_eval_cfg = dict(evaluator=dict(type=GPQAEvaluator),
#                      pred_postprocessor=dict(type=math_postprocess_v2))

gpqa_datasets = []
gpqa_subsets = {
    # 'extended': 'gpqa_extended.csv',
    # 'main': 'gpqa_main.csv',
    'diamond': 'gpqa_diamond.csv'
}

for split in list(gpqa_subsets.keys()):
    gpqa_datasets.append(
        dict(
            abbr='GPQA_' + split,
            type=GPQADataset,
            path='./data/gpqa/',
            name=gpqa_subsets[split],
            reader_cfg=gpqa_reader_cfg,
            infer_cfg=gpqa_infer_cfg,
            eval_cfg=gpqa_eval_cfg)
    )
