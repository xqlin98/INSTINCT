export export OPENAI_API_KEY=YOUR_KEY
datasets=(gsm8k)
for task in ${datasets[@]}; do
    python experiments/run_cot_bandits.py \
    --task ${task} \
    --n_prompt_tokens 10 \
    --nu 1 \
    --lamdba 0.1 \
    --n_init 40 \
    --n_domain 10000 \
    --total_iter 165 \
    --local_training_iter 1000 \
    --n_eval 1000 \
    --intrinsic_dim 1000 \
    --gpt gpt-3.5-turbo-0301 \
    --name iter165_gpt-0301
done