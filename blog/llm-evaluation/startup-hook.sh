#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt

git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
git checkout e81d3cce155e93ba2445068767c738891ad97024
pip install -e ".[multilingual, sentencepiece, anthropic]"
# Specific package required for TruthfulQAGeneration task.
# See https://github.com/EleutherAI/lm-evaluation-harness/blob/e81d3cce155e93ba2445068767c738891ad97024/lm_eval/tasks/truthfulqa.py#L176-L180
pip install bleurt@https://github.com/google-research/bleurt/archive/b610120347ef22b494b6d69b4316e303f5932516.zip#egg=bleurt
cd ..