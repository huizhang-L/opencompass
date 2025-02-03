import concurrent.futures
import json
import re

from datasets import Dataset

from opencompass.models import OpenAISDK
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET, MODELS

from .base import BaseDataset

# from opencompass.utils import get_data_path

EVAL_PROMPT = """
请你作为一个数学高考阅卷专家，判断下面的答案是否与标准答案一致，即考生是否回答正确。下面是一些评判标准：
1. 有些答案可能包含多项内容，可能有单选题，多选题，填空题等，只要答案与标准答案一致即可, 对于多选题和多个空的填空题，需要考生对应的选项或空都回答正确才算正确。
2. 有些答案可能通过不同的方式表达，比如有些答案可能是一个数学表达式，有些答案可能是一个文字描述，只要表达的意思一致即可。且有些公式通过不同的方式表达，但等价，也是正确的。
3. 你不需要重新计算问题答案，因为标准答案已经给出，只需要根据问题形式来判断考生的答案是否与标准答案一致，是否正确即可。

请你根据上述标准，判断下面的答案是否与标准答案一致，如果一致，请在最后输出\\boxed{{yes}}, 否则输出\\boxed{{no}}, 如果难以判断，请输出\\boxed{{no}}.
原问题：{question}
标准答案：{gold_answer}
考生答案：{answer}

分析：
""" # noqa E501

POST_PROMPT_CN="""
你是一个乐于助人的助手，任务是从给定的回答句子中提取精确的关键答案。你必须只提供提取的关键答案，不包括任何额外的文字。
—
我将为你提供一个问题、回答句子和问题类型。回答句子是对所提供问题的回应。利用提供的信息，你必须准确而精确地确定并从回答句子中提取预期的关键答案。请不要对问题发表主观看法。

对于单选题，答案应该是选项字母，例如 "A"；
对于多选题，答案应该是一个选项字母的列表，例如 ["A"] 或 ["A", "B", "C"]；
对于填空题，答案应该是一个填入空白处的答案列表，列表的数量应该与问题中的空白数量相同，同一空白的答案可能有多个，请在同一个 string 中用逗号隔开表示，如 ['sqrt(x) 且 x > 10', '1/2, 1/3', '1/4'] 代表问题包含三小问，第一小问包含取值范围信息，第二小问有两个答案，第三小问有一个答案。
对于解答题，类似填空题，答案应该是一个答案列表，每小问的答案间用逗号隔开，同样需要注意某些小问答案多个的情况。

如果回答句子提供了多个不同的答案，请仔细判断后面提供的答案是否是对前面答案的修正或修改。如果是这样，提取这个修正或修改后的答案作为最终答案。相反，如果回答句子在多个答案之间波动而没有明确的最终答案，你应该输出 [No valid answer]。
—
问题类型: {question_type}
原始问题: {question}
回答: {response}
提取的关键答案:
""" # noqa E501

POST_PROMPT_EN="""
You are a helpful assistant whose task is to extract precise key answers from given response sentences. You must only provide the extracted key answers without any additional text.
—
I will provide you with a question, a response sentence, and the question type. The response sentence is a reply to the provided question. Using the provided information, you must accurately and precisely identify and extract the expected key answers from the response sentence. Please do not provide subjective opinions about the question.

For multiple-choice questions, the answer should be the letter of the option, such as "A".
For multiple-answer questions, the answer should be a list of option letters, such as ["A"] or ["A", "B", "C"].
For fill-in-the-blank questions, the answer should be a list of answers to fill in the blanks. The number of items in the list should match the number of blanks in the question. If there are multiple answers for the same blank, separate them with a comma within the same string, like ['sqrt(x) and x > 10', '1/2, 1/3', '1/4'], which represents three sub-questions where the first sub-question includes a range, the second sub-question has two answers, and the third sub-question has one answer.
For problem-solving questions, similar to fill-in-the-blank questions, the answer should be a list of answers. Separate answers for different sub-questions with commas, and note that some sub-questions may have multiple answers.

If the response sentence provides multiple different answers, carefully determine whether a later provided answer is a correction or modification of an earlier answer. If so, extract this corrected or modified answer as the final answer. Conversely, if the response sentence fluctuates between multiple answers without a clear final answer, you should output [No valid answer].
—
Question type: {question_type}
Question: {question}
Output sentences: {response}
Key extracted answer:
""" # noqa E501
GRADER_TEMPLATE="""
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 

Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the question is multiple choice, don't focus too much on the form of the answer. The answer can be an option or a specific answer or other reasonable form.

Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT 
B: INCORRECT
Just return the letters "A" or "B", with no text around it.

Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


<Original Question Begin>: \n{question}\n<Original Question End>\n\n
<Gold Target Begin>: \n{gold_answer}\n<Gold Target End>\n\n
<Predicted Answer Begin>: \n{answer}\n<Predicted End>\n\n

Judging the correctness of candidates' answers:
""".strip()

# Meta_Instruction = """You are a help assistant tasked with extracting the precise key answer from given output sentences. You must only provide the extracted key answer without including any additional text.
# -
# I will provide you with a question, output sentences along with an answer range. The output sentences are the response of the question provided. The answer range could either describe the type of answer expected or list all possible valid answers. Using the information provided, you must accurately and precisely determine and extract the intended key answer from the output sentences. Please don't have your subjective thoughts about the question.
# First, you need to determine whether the content of the output sentences is relevant to the given question. If the entire output sentences are unrelated to the question (meaning the output sentences are not addressing the question), then output [No valid answer].
# Otherwise, ignore the parts of the output sentences that have no relevance to the question and then extract the key answer that matches the answer range.
# Below are some special cases you need to be aware of:
#     (1) If the answer range is a list and the key answer in the output sentences is not explicitly listed among the candidate options in the answer range, also output [No valid answer].
#     (2) You should only return the precise answer you extract, without processing the answer. Please return only the answer and do not add any additional content.
#     (3) If the response sentence provides multiple different answers, carefully determine whether a later provided answer is a correction or modification of an earlier answer. If so, extract this corrected or modified answer as the final answer. Conversely, if the response sentence fluctuates between multiple answers without a clear final answer, you should output [No valid answer].
# —
# Question: {question}
# Output sentences: {response}
# Key extracted answer:
# """ # noqa
Meta_Instruction = """You are a help assistant tasked with extracting the precise key answer from given output sentences. You must only provide the extracted key answer without including any additional text.
-
I will provide you with a question, output sentences along with an answer range. The output sentence is the answer or the final part of the answer to the provided question. The answer range could either describe the type of answer expected or list all possible valid answers. Using the information provided, you must accurately and precisely determine and extract the intended key answer from the output sentences. Please don't have your subjective thoughts about the question.
First, you need to determine whether the content of the output sentences is relevant to the given question. If the entire output sentences are unrelated to the question (meaning the output sentences are not addressing the question), then output [No valid answer].
Otherwise, ignore the parts of the output sentences that have no relevance to the question and then extract the key answer that matches the answer range.
Below are some special cases you need to be aware of:
    (1) If the answer range is a list and the key answer in the output sentences is not explicitly listed among the candidate options in the answer range, also output [No valid answer].
    (2) You should only return the precise answer you extract, without processing the answer. Please return only the answer and do not add any additional content.
    (3) If the response sentence provides multiple different answers, carefully determine whether a later provided answer is a correction or modification of an earlier answer. If so, extract this corrected or modified answer as the final answer. Conversely, if the response sentence fluctuates between multiple answers without a clear final answer, you should output [No valid answer].
—
Question: {question}
Output sentences: {response}
Key extracted answer:
""" # noqa

# def extract_boxed_answer(text):
#     match = re.findall(r'\\boxed{(.+?)}', text)
#     if match:
#         return match[-1]
#     return None

def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None


def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer

def extract_boxed_answer_new(text):
    # 去除开头的空格
    stripped_answer = text.lstrip()
    # 检查是否以 'A' 开头
    return stripped_answer.startswith('A')

def extract_last_n_sentences(text, n=5):
    text = text.strip()
    if text and text[-1] not in ".!?":  # 如果最后没有标点符号，则加上 "."
        text += "."

    # 按句号（.）、问号（?）、感叹号（!）分割句子
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # 取最后 n 句话
    last_sentences = sentences[-n:]

    # 重新拼接成字符串
    return ' '.join(last_sentences)

@LOAD_DATASET.register_module()
class GaoKaoMATHDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        # path = get_data_path(path, local_mode=True)
        data = json.load(open(path))
        for i in range(len(data)):
            data[i]['extract_answer'] = str(data[i]['extract_answer'])
        dataset = Dataset.from_list(data)
        return dataset


api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])


@ICL_EVALUATORS.register_module()
class GaoKaoMATHEvaluator(BaseEvaluator):

    def __init__(self,
                 model_name,
                 url,
                 question_type=None,
                 language='en',
                 with_postprocess=False,
                 post_url=[],
                 post_model_name='',
                 **kwargs):
        if isinstance(url, str):
            url = [url]

        self.model = [
            MODELS.build(
                dict(
                    type=OpenAISDK,
                    path=model_name,
                    openai_api_base=url,
                    key='EMPTY',
                    query_per_second=16,
                    meta_template=api_meta_template,
                    temperature=kwargs.get('temperature', 1e-6),
                    max_seq_len=kwargs.get('max_tokens', 30000),
                )) for url in url
        ]
        self.question_type = question_type
        self.language = language
        self.with_postprocess = with_postprocess
        self.post_url = post_url
        self.post_model_name = post_model_name

    def batch_response(self, models, inputs):
        batch_num = len(models)
        batch_size = (len(inputs) + batch_num - 1) // batch_num
        result_responses = []

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=batch_num) as executor:
            futures = [
                executor.submit(models[i].generate,
                                inputs[i * batch_size:(i + 1) * batch_size])
                for i in range(batch_num)
            ]
            for response in executor.map(lambda f: f.result(), futures):
                result_responses.extend(response)
        return result_responses

    def postprocess(self, questions, predictions, question_type='None'):
        self.post_model = [
            MODELS.build(
                dict(
                    type=OpenAISDK,
                    path=self.post_model_name,
                    openai_api_base=url,
                    key='EMPTY',
                    query_per_second=16,
                    meta_template=api_meta_template,
                    temperature=1e-6,
                    max_seq_len=30000,
                )) for url in self.post_url
        ]
        input_prompts = []
        rule_extracted_answer = []
        extracted_answer = []
        # prompt = POST_PROMPT_EN if self.language == 'en' else POST_PROMPT_CN
        # modified by lhj
        prompt = Meta_Instruction if self.language == 'en' else POST_PROMPT_CN
        for question, response, question_type in zip(questions, predictions,
                                                     question_type):
            # input_prompts.append(
            #     prompt.format(question=question,
            #                   response=response,
            #                   question_type=question_type))
            # added by lhj
            cand_ans = extract_boxed_answer(response)
            last_response = extract_last_n_sentences(response, 5)
            if cand_ans:
                rule_extracted_answer.append(cand_ans)
                input_prompts.append(
                    prompt.format(question=question,
                                response=last_response))
            else:
                rule_extracted_answer.append('')
                input_prompts.append(
                    prompt.format(question=question,
                                response=last_response))
        result_responses = self.batch_response(self.post_model, input_prompts)
        # added by lhj
        for  rea, rr in zip(rule_extracted_answer, result_responses):
            if rea != '':
                extracted_answer.append(rea)
            else:
                extracted_answer.append(rr)
        return extracted_answer

    def score(self, predictions, references, origin_prompt, test_set):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        questions = [item[0]['prompt'] for item in origin_prompt]
        count = 0
        correct = 0
        details = []
        results = []

        if self.with_postprocess:
            if self.question_type:
                self.question_type = [self.question_type] * len(questions)
            # test_set type is huggingface Dataset
            elif 'question_type' in test_set.column_names:
                self.question_type = test_set['question_type']
            else:
                self.question_type = ['问答题'] * len(
                    questions) if self.language == 'cn' else [
                        'problem-solving'
                    ] * len(questions)

            predictions = self.postprocess(questions, predictions,
                                           self.question_type)

        inputs = []
        for pred, ref, ques in zip(predictions, references, questions):
            # inputs.append(
            #     EVAL_PROMPT.format(answer=pred, gold_answer=ref,
            #                        question=ques))
            # modified by lhj
            inputs.append(
                GRADER_TEMPLATE.format(answer=pred, gold_answer=ref,
                                   question=ques))
        result_responses = self.batch_response(self.model, inputs)

        results = [
            extract_boxed_answer_new(result)
            for result in result_responses
        ]
        for pred, ref, result, result_response in zip(predictions, references,
                                                      results,
                                                      result_responses):
            detail = {
                'pred': pred,
                'answer': ref,
                'correct': False,
                'eval_model_response': result_response
            }
            count += 1
            if result:
                correct += 1
                detail['correct'] = True
            details.append(detail)

        detailed_result = {
            'accuracy': 100 * correct / count,
            'details': details
        }

        return detailed_result
