# Use Your INSTINCT: INSTruction optimization for LLMs usIng Neural bandits Coupled with Transformers [ICML 2024]
Xiaoqiang Lin*, Zhaoxuan Wu*, Zhongxiang Dai, Wenyang Hu, Yao Shu, See-Kiong Ng, Patrick Jaillet, Bryan Kian Hsiang Low

[Project Homepage](https://xqlin98.github.io/INSTINCT/) | [ArXiv](https://arxiv.org/abs/2310.02905) | [Paper](https://arxiv.org/pdf/2310.02905.pdf)

This is the code for the paper: Use Your INSTINCT: Instruction Optimization Using Neural Bandits Coupled with Transformers.
We provide all the codes for our experiments which includes:
- Instruction induction
- Improving chain-of-thought instruction

Our code are based on the code from [APE](https://github.com/keirp/automatic_prompt_engineer) and [InstructZero](https://github.com/Lichang-Chen/InstructZero).

# Abstract
Large language models (LLMs) have shown remarkable instruction-following capabilities and achieved impressive performances in various applications. However, the performances of LLMs depend heavily on the instructions given to them, which are typically manually tuned with substantial human efforts. Recent work has used the query-efficient Bayesian optimization (BO) algorithm to automatically optimize the instructions given to black-box LLMs. However, BO usually falls short when optimizing highly sophisticated (e.g., high-dimensional) objective functions, such as the functions mapping an instruction to the performance of an LLM. This is mainly due to the limited expressive power of the Gaussian process (GP) model which is used by BO as a surrogate to model the objective function. Meanwhile, it has been repeatedly shown that neural networks (NNs), especially pre-trained transformers, possess strong expressive power and can model highly complex functions. So, we adopt a neural bandit algorithm which replaces the GP in BO by an NN surrogate to optimize instructions for black-box LLMs. More importantly, the neural bandit algorithm allows us to naturally couple the NN surrogate with the hidden representation learned by a pre-trained transformer (i.e., an open-source LLM), which significantly boosts its performance. These motivate us to propose our INSTruction optimization usIng Neural bandits Coupled with Transformers (INSTINCT) algorithm. We perform instruction optimization for ChatGPT and use extensive experiments to show that our INSTINCT consistently outperforms the existing methods in different tasks, such as in various instruction induction tasks and the task of improving the zero-shot chain-of-thought instruction.
# Prepare the data
You can download the data for intrinsic induction from the github repo of [InstructZero](https://github.com/Lichang-Chen/InstructZero). You can download the dataset of [SAMSum](https://huggingface.co/datasets/samsum) from the huggingface website. You can download the dataset for GSM8K, AQUARAT, and SVAMP from the repo for [APE](https://github.com/keirp/automatic_prompt_engineer).

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
Note that before you run the above bash script, you need to specify the openai key for calling `gpt-turbo-3.5-0301` API. To do so, change the following in the two bash scripts:
```
export export OPENAI_API_KEY=YOUR_KEY
```

## BibTeX
```
@inproceedings{lin2024use,
        title={Use Your {INSTINCT}: INSTruction optimization for LLMs usIng Neural bandits Coupled with Transformers},
        author={Xiaoqiang Lin and Zhaoxuan Wu and Zhongxiang Dai and Wenyang Hu and Yao Shu and See-Kiong Ng and Patrick Jaillet and Bryan Kian Hsiang Low},
        year={2024},
        booktitle={Proc. ICML}
}
```
