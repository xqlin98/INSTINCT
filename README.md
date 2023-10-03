This is the code for the paper: Use Your INSTINCT: Instruction Optimization Using Neural Bandits Coupled with Transformers.
We provide all the codes for our experiments which includes:
- Instruction induction
- Improving chain-of-thought instruction
Our code are based on the code from APE (https://github.com/keirp/automatic_prompt_engineer) and InstructZero (https://github.com/Lichang-Chen/InstructZero).
# Prepare the data
You can download the data for intrinsic induction from the github repo of InstructZero: https://github.com/Lichang-Chen/InstructZero. You can download the dataset of SAMSum from the huggingface website: https://huggingface.co/datasets/samsum. You can download the dataset for GSM8K, AQUARAT, and SVAMP from the repo for APE: https://github.com/keirp/automatic_prompt_engineer.

We put the data preparsion notebook at `COT/experiments/data/instruction_induction/pre_aqua.ipynb`, `COT/experiments/data/instruction_induction/pre_gsm8k.ipynb` and `Induction/experiments/data/nlptasks/pre_nlp_data.ipynb`.

# Run our code
To run our code, you need to install the environment using conda:
`conda env create -f environment.yml`

We provide bash scripts for running our experiments for instruction induction at `Induction/experiments/run_neural_bandits.sh`. To run it properly, you need to run the following in the terminal:
```
cd Induction
bash experiments/run_neural_bandits.sh
```
Similarly, to run our code for improving chain-of-thought instruction, you need to run the script `COT/experiments/run_cot_bandits.sh` as the following:
```
cd COT
bash experiments/run_cot_bandits.sh
```