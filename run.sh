#!/bin/bash

for i in {1..10}
do
    python run.py configs/eval_math_llmpostprocess_llmjudge_4.py --dump-eval-details
    sleep 5
done