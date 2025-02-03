# vllm serve
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /opt/project/n/mm_school/fudan_extra_storage/lhj/model/QwQ-32B-Preview/QwQ-32B-Preview --host 0.0.0.0 --port 37111 --served-model-name qwq-32b-preview --tensor-parallel-size 8

# 4卡 荣耀
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /opt/project/n/mm_school/fudan_extra_storage/lhj/model/QwQ-32B-Preview/QwQ-32B-Preview --host 0.0.0.0 --port 37111 --served-model-name qwq-32b-preview --tensor-parallel-size 4
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /opt/project/n/mm_school/fudan_extra_storage/lhj/model/Qwen2.5-32B-Instruct/Qwen2.5-32B-Instruct --host 0.0.0.0 --port 37112 --served-model-name qwen2.5-32b-instruct --tensor-parallel-size 4
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /opt/project/n/mm_school/fudan_extra_storage/lhj/model/merge_model/qwq_9_qwen_1 --host 0.0.0.0 --port 37113 --served-model-name qwq-qwen2.5-merge --tensor-parallel-size 4
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /opt/project/n/mm_school/fudan_extra_storage/lhj/model/Llama-3.3-70B-Instruct --host 0.0.0.0 --port 37114 --served-model-name llama3.3-70b-instruct --tensor-parallel-size 4

# 2卡
CUDA_VISIBLE_DEVICES=0,1 vllm serve /opt/project/n/mm_school/fudan_extra_storage/lhj/model/QwQ-32B-Preview/QwQ-32B-Preview --host 0.0.0.0 --port 37111 --served-model-name qwq-32b-preview --tensor-parallel-size 2
CUDA_VISIBLE_DEVICES=2,3 vllm serve /opt/project/n/mm_school/fudan_extra_storage/lhj/model/Qwen2.5-32B-Instruct/Qwen2.5-32B-Instruct --host 0.0.0.0 --port 37112 --served-model-name qwen2.5-32b-instruct --tensor-parallel-size 2
CUDA_VISIBLE_DEVICES=0,1 vllm serve /mnt/data/user/lv_huijie/_MODELS/qwq-qwen2.5-32b-instruct-merge --trust-remote-code --host 0.0.0.0 --port 37113 --served-model-name qwq-qwen2.5-merge --tensor-parallel-size 2
# 调试1卡
CUDA_VISIBLE_DEVICES=7 vllm serve /mnt/data/models/pretrain_models/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 37115 --served-model-name qwen2.5-7b-instruct --tensor-parallel-size 1
# run eval
python run.py configs/eval_math_llmpostprocess_llmjudge_0.py --dump-eval-details
python run.py configs/eval_math_llmpostprocess_llmjudge_1.py --dump-eval-details
python run.py configs/eval_math_llmpostprocess_llmjudge_2.py --dump-eval-details
python run.py configs/eval_math_llmpostprocess_llmjudge_3.py --dump-eval-details
python run.py configs/eval_math_llmpostprocess_llmjudge_4.py --dump-eval-details

# check prompt
python tools/prompt_viewer.py opencompass/configs/datasets/gsm8k/gsm8k_0shot_nocot_gen_6cbf22_test.py -n
python tools/prompt_viewer.py configs/eval_vllm_serve_test.py -n


CUDA_VISIBLE_DEVICES=1,2 python token_count.py

"You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."
"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
curl http://localhost:37114/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3.3-70b-instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'CUDA_VISIBLE_DEVICES=0,1 vllm serve /opt/project/n/mm_school/fudan_extra_storage/lhj/model/QwQ-32B-Preview/QwQ-32B-Preview --host 0.0.0.0 --port 37111 --served-model-name qwq-32b-preview --tensor-parallel-size 2

# merge model
mergekit-yaml /opt/project/n/mm_school/fudan_extra_storage/lhj/mergekit/examples/linear1.yml /opt/project/n/mm_school/fudan_extra_storage/lhj/model/merge_model/qwq_1_qwen_9 --cuda
mergekit-yaml /opt/project/n/mm_school/fudan_extra_storage/lhj/mergekit/examples/linear2.yml /opt/project/n/mm_school/fudan_extra_storage/lhj/model/merge_model/qwq_2_qwen_8 --cuda
mergekit-yaml /opt/project/n/mm_school/fudan_extra_storage/lhj/mergekit/examples/linear3.yml /opt/project/n/mm_school/fudan_extra_storage/lhj/model/merge_model/qwq_3_qwen_7 --cuda
mergekit-yaml /opt/project/n/mm_school/fudan_extra_storage/lhj/mergekit/examples/linear4.yml /opt/project/n/mm_school/fudan_extra_storage/lhj/model/merge_model/qwq_4_qwen_6 --cuda
mergekit-yaml /opt/project/n/mm_school/fudan_extra_storage/lhj/mergekit/examples/linear5.yml /opt/project/n/mm_school/fudan_extra_storage/lhj/model/merge_model/qwq_5_qwen_5 --cuda
mergekit-yaml /opt/project/n/mm_school/fudan_extra_storage/lhj/mergekit/examples/linear6.yml /opt/project/n/mm_school/fudan_extra_storage/lhj/model/merge_model/qwq_6_qwen_4 --cuda
mergekit-yaml /opt/project/n/mm_school/fudan_extra_storage/lhj/mergekit/examples/linear7.yml /opt/project/n/mm_school/fudan_extra_storage/lhj/model/merge_model/qwq_7_qwen_3 --cuda
mergekit-yaml /opt/project/n/mm_school/fudan_extra_storage/lhj/mergekit/examples/linear8.yml /opt/project/n/mm_school/fudan_extra_storage/lhj/model/merge_model/qwq_8_qwen_2 --cuda
mergekit-yaml /opt/project/n/mm_school/fudan_extra_storage/lhj/mergekit/examples/linear9.yml /opt/project/n/mm_school/fudan_extra_storage/lhj/model/merge_model/qwq_9_qwen_1 --cuda

cp /opt/project/n/mm_school/fudan_extra_storage/lhj/model/QwQ-32B-Preview/QwQ-32B-Preview/generation_config.json /opt/project/n/mm_school/fudan_extra_storage/lhj/model/merge_model/qwq_5_qwen_5
cp /opt/project/n/mm_school/fudan_extra_storage/lhj/model/QwQ-32B-Preview/QwQ-32B-Preview/vocab.json /opt/project/n/mm_school/fudan_extra_storage/lhj/model/merge_model/qwq_5_qwen_5


# 看实例里所有卡的使用情况
vc -gpu list
# 看到节点名称
vc -node list
# 免密登录到其他节点上
vc -ssh mm-mllm-gui-fd-0
vc -ssh mm-mllm-gui-fd-1
vc -ssh mm-mllm-gui-fd-2
vc -ssh mm-mllm-gui-fd-3
vc -ssh mm-mllm-gui-fd-4
# 杀占卡
ps -ef | grep train_gpu | awk '{print $2}' | xargs kill -9