import random
import torch
import numpy as np
import copy
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
from automatic_prompt_engineer import ape, data
from experiments.data.instruction_induction.load_data import load_data
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator, exec_evaluator
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, LlamaForCausalLM, BertForSequenceClassification
from LlamaForMLPRegression import LlamaForMLPRegression, MLPRegression, MLPRegression_Train, NeuralTSDiag
from automatic_prompt_engineer import evaluate, config, template, data
import os
import re

from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
# from instruction_coupled_kernel import *
from tqdm import tqdm
import argparse
from experiments.evaluation.instruction_induction.utility import set_all_seed, TASKS
import datetime

SMOKE_TEST = os.environ.get("SMOKE_TEST")
## bayesian opt
tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}
    
    
# Evaluation budget
# N_DOMAIN=1000
# N_INIT=5
# TOTAL_ITER = 120
# STOP_TRAINING_AFTER_ITER=2000
# LOCAL_TRAINING_ITER=30
# N_ITERATIONS = 4 if not SMOKE_TEST else 1
# BATCH_SIZE = 20 if not SMOKE_TEST else 1
# print(f"Using a total of {TOTAL_ITER} function evaluations")

model_name = "vicuna"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_model = 'chatgpt'
alpha = 1
sigma = 1

class LMForwardAPI:
    def __init__(self, model_name='vicuna', eval_data=None, init_prompt=None, init_qa=None, conf=None, base_conf=None,
                 prompt_gen_data=None, n_prompt_tokens=None, few_shot_data=None, 
                 HF_cache_dir=None, random_proj=None, intrinsic_dim=None):
        p = torch.ones(10)
        
        kwargs={'torch_dtype': torch.float16}
        if model_name in ["vicuna", "alpaca", "flan-t5"]:
            self.model = LlamaForMLPRegression.from_pretrained(
                                HF_cache_dir, low_cpu_mem_usage=True, **kwargs
                            ).cuda()

            self.tokenizer = AutoTokenizer.from_pretrained(
                                HF_cache_dir,
                                model_max_length=512,
                                padding_side='left',
                                use_fast=False,
                            )
            
        else:
            raise NotImplementedError

        self.init_token = init_prompt[0] + init_qa[0]
        if model_name in ['alpaca', 'vicuna']:
            self.embedding = self.model.get_input_embeddings().weight.clone()
            input_ids = self.tokenizer(init_prompt, return_tensors="pt").input_ids.cuda()
            self.init_prompt = self.embedding[input_ids]
            
        ################# setup n_prompts_token #################
        self.n_prompt_tokens = n_prompt_tokens
        self.hidden_size = self.init_prompt.shape[-1]
        print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))
        
        # self.init_prompt = self.init_prompt.reshape(self.n_prompt_tokens * self.hidden_size)
        self.count = 0
        self.linear = torch.nn.Linear(intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False)
        
        if random_proj == 'normal':
            # calculate std for normal distribution
            if model_name in ['llama', 'alpaca', 'vicuna']:
                print('Get the embedding firstly to avoid issues')
            else:
                raise NotImplementedError
            mu_hat = np.mean(self.embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(self.embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = alpha * std_hat / (np.sqrt(intrinsic_dim) * sigma)

            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear.parameters():   
                torch.nn.init.uniform_(p, -1, 1)
        elif random_proj == 'uniform':  
            for p in self.linear.parameters():   
                torch.nn.init.uniform_(p, -1, 1)
                
        ## eval preparation
        self.conf = config.update_config(conf, base_conf)
        self.eval_data = eval_data
        self.eval_template = template.EvalTemplate("Instruction: [PROMPT]\n\nInput: [INPUT]\n Output: [OUTPUT]")
        self.demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")
        
        if api_model in ['llama', 'flan-t5']:
            self.api_model = exec_evaluator(api_model, self.conf)

        if few_shot_data is None:
            self.few_shot_data = prompt_gen_data
        
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()
        self.prompts_list = []

    def get_last_token_hidden_state(self, prompt_embedding):
        
        input_ids = self.tokenizer(self.init_token, return_tensors="pt").input_ids.cuda()
        input_embed = self.embedding[input_ids]
        prompt_embedding_ = prompt_embedding.to(device=input_embed.device, dtype=input_embed.dtype).reshape(1, self.n_prompt_tokens, -1)
        input_embed = torch.cat((prompt_embedding_, input_embed), 1)
        last_token_id = input_embed.shape[1] - 1
        # last_token_id = 0
        hidden_state, = self.model.get_last_token_hidden_state(inputs_embeds=input_embed, sequence_lengths=last_token_id)        
        return hidden_state
    

    def get_last_token_hidden_state_batch(self, prompt_embedding, pooling='last'):
        size = prompt_embedding.shape[0]
        input_ids = self.tokenizer(self.init_token, return_tensors="pt").input_ids.cuda()
        
        batch_size = 10
        n_batchs = size // batch_size + int((size % batch_size) != 0)
        all_hidden_state = []
        for i in tqdm(range(n_batchs), desc='Get hidden states'):
            if i == n_batchs - 1:
                prompt_batch = prompt_embedding[(i*batch_size):]
            else:
                prompt_batch = prompt_embedding[(i*batch_size):((i+1)*batch_size)]
            batch_size_ = prompt_batch.shape[0]
            input_embed = self.embedding[input_ids]
            input_embed = input_embed.repeat(batch_size_, 1, 1)
            prompt_embedding_ = prompt_batch.to(device=input_embed.device, dtype=input_embed.dtype).reshape(batch_size_, self.n_prompt_tokens, -1)
            input_embed = torch.cat((prompt_embedding_, input_embed), 1)
            last_token_id = input_embed.shape[1] - 1
            
            hidden_state_, = self.model.get_last_token_hidden_state(inputs_embeds=input_embed, sequence_lengths=last_token_id, pooling=pooling)
            all_hidden_state.append(hidden_state_)
        
        all_hidden_state = torch.vstack(all_hidden_state)        
        return all_hidden_state
    
    def eval(self, prompt_embedding=None, test_data=None):
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        tmp_prompt = prompt_embedding.detach().clone()  # list or numpy.ndarray
        if isinstance(prompt_embedding, list):  # multiple queries
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim
    
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            # prompt_embedding = self.linear(prompt_embedding)  # Az
            # if self.init_prompt is not None:
            #     prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        elif isinstance(prompt_embedding, torch.Tensor): 
            prompt_embedding = prompt_embedding.type(torch.float32)
            # prompt_embedding = self.linear(prompt_embedding)  # Az
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )

        input_ids = self.tokenizer(self.init_token, return_tensors="pt").input_ids.cuda()
        input_embed = self.embedding[input_ids]
        prompt_embedding = prompt_embedding.to(device=input_embed.device, dtype=input_embed.dtype)
        input_embed = torch.cat((prompt_embedding, input_embed), 1)
        outputs = self.model.generate(inputs_embeds=input_embed, max_new_tokens=64)
        instruction = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # postprocess instruction
        instruction[0] = 'The instruction was to ' + instruction[0]
        start = instruction[0].find('The instruction was to')
        end = instruction[0].find('Comment:')
        if end == -1:
            instruction[0] = instruction[0][start:]
        else:
            instruction[0] = instruction[0][start: end]

        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', instruction[0])
        search_string = 'The instruction was to'
        for sentence in sentences:
            if sentence.startswith(search_string):
                instruction[0] = sentence
                break

        # print post-processed instruction
        print('Instruction: {}'.format(instruction))
        
        if instruction[0] in self.prompts_set.keys():
            (dev_perf, instruction_score) = self.prompts_set[instruction[0]]
        else:
            if api_model in ['chatgpt']: 
                dev_perf, instruction_score = evaluate.evaluate_prompts(instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data, self.conf['evaluation']['method'], self.conf['evaluation'])
                dev_perf = dev_perf.sorted()[1][0]
                self.prompts_set[instruction[0]] = (dev_perf, instruction_score)
            # We will fix the bugs for other api models. Stay tuned!
            # elif api_model in ['llama', 'flan-t5']: 
            #     dev_perf, instruction_score = self.api_model.evaluate(instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data,
            #                             self.conf['evaluation']).sorted()[1][0]            
            #     self.prompts_set[instruction[0]] = (dev_perf, instruction_score)
            else:
                raise NotImplementedError
        self.prompts_list.append((len(self.prompts_list), instruction[0], dev_perf))
        if dev_perf >= self.best_last_perf:
            self.count += 1

        if dev_perf >= self.best_dev_perf:
            self.best_dev_perf = dev_perf
            self.best_prompt = copy.deepcopy(tmp_prompt)
            self.best_instruction = instruction

        print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
            round(float(dev_perf), 4),
            round(float(dev_perf), 4),
            round(float(self.best_dev_perf), 4)))
        print('********* Done *********')

        return dev_perf, instruction_score

    def return_best_prompt(self):
        return self.best_instruction

    def return_prompts_set(self):
        return self.prompts_set

    def return_prompts_list(self):
        return self.prompts_list
    
def run(task, n_prompt_tokens, HF_cache_dir, nu, lamdba, n_init, n_domain, total_iter, local_training_iter, random_proj, intrinsic_dim, n_eval, gpt, init_scale, pooling):
    assert task in TASKS, 'Task not found!'

    induce_data, test_data = load_data('induce', task), load_data('eval', task)

    # Get size of the induce data
    induce_data_size = len(induce_data[0])
    prompt_gen_size = min(int(induce_data_size * 0.5), 100)
    # Induce data is split into prompt_gen_data and eval_data
    prompt_gen_data, eval_data = data.create_split(
        induce_data, prompt_gen_size)

    # Data is in the form input: single item, output: list of items
    # For prompt_gen_data, sample a single item from the output list
    prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0]
                                           for output in prompt_gen_data[1]]

    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\n\nOUTPUT: [OUTPUT]" # change the evaluation template
    init_prompt = ['\n']
    prompt_gen_template = "[full_DEMO]\n\nThe instruction was to"

    base_conf = '../experiments/configs/instruction_induction.yaml'
    conf = {
        'generation': {
            'num_subsamples': 1,
            'num_demos': 5,
            'num_prompts_per_subsample': 20,
            'model': {
                'gpt_config': {
                    'model': gpt
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': min(20, len(eval_data[0])),
            'model': {
                'gpt_config': {
                    'model': gpt
                }
            }
        }
    }

    # make the demo automatically
    subsampled_data = data.subsample_data(prompt_gen_data, conf['generation']['num_demos'])
    prompt_gen_template = template.InitQATemplate(prompt_gen_template)
    d_template = template.DemosTemplate(demos_template)
    demos = d_template.fill(subsampled_data)
    init_qa = [prompt_gen_template.fill(demos)]

    model_forward_api = LMForwardAPI(model_name=model_name, eval_data=eval_data, init_prompt=init_prompt, 
                                    init_qa=init_qa, conf=conf, base_conf=base_conf, prompt_gen_data=prompt_gen_data,
                                    n_prompt_tokens=n_prompt_tokens, HF_cache_dir=HF_cache_dir, random_proj=random_proj,intrinsic_dim=intrinsic_dim)
    
        
    # start bayesian opt
    all_X = SobolEngine(dimension=intrinsic_dim, scramble=True, seed=0).draw(n_domain)
    # all_X = model_forward_api.linear(all_X)/np.sqrt(intrinsic_dim/10)
    all_X = model_forward_api.linear(all_X)

    print("end engine")
    init_idxs = np.random.choice(n_domain, n_init, replace=False)
    # scales = torch.linspace(0.01, init_scale, steps=n_init)
    # all_X[init_idxs] = all_X[init_idxs] * scales.reshape(-1, 1)

    all_X_mlp = model_forward_api.get_last_token_hidden_state_batch(all_X, pooling=pooling).to(**tkwargs)

    X = all_X[init_idxs]
    
    X_return = [model_forward_api.eval(x) for x in X]
    X_mlp = all_X_mlp[init_idxs]
    Y = [x[0] for x in X_return]
    
    X = X.to(**tkwargs)
    Y = torch.FloatTensor(Y).unsqueeze(-1).to(**tkwargs)
    X_mlp = X_mlp.to(**tkwargs)
    print(f"Best initial point: {Y.max().item():.3f}")


    # standardization Y (no standardization for X)
    X_train = X_mlp
    y_train = Y
    torch.save({"X_train": X_train, "y_train": y_train, "all_X_mlp": all_X_mlp}, f'train_{task}.pt')
    # y_train = (Y - Y.mean(dim=-2))/(Y.std(dim=-2))

    max_iter = total_iter - n_init
    context = all_X_mlp
    
    # lambda represent regularization, nu represent the trade-off, grid search
    l = NeuralTSDiag(input_dim=model_forward_api.hidden_size, lamdba=lamdba, nu=nu, init_x=X_train, init_y=y_train, style='ucb', diagonalize=True)
    best_r = 0
    print("Max iter: ", max_iter)
    best_values = []
    for t in range(max_iter):
        print("Start selecting...")
        
        # randomly select n_eval points to evaluate
        if n_domain != n_eval:
            selected_idx = np.random.choice(n_domain, n_eval, replace=False)
            arm_select, _ = l.select(context[selected_idx])
            # arm_select, _ = l.select(context)
            r = model_forward_api.eval(all_X[selected_idx][arm_select])[0]
            best_r = max(r, best_r)
            print("Start training...")
            l.train(context[selected_idx][arm_select], r, local_training_iter)
        else:
            arm_select, _ = l.select(context)
            # arm_select, _ = l.select(context)
            r = model_forward_api.eval(all_X[arm_select])[0]
            best_r = max(r, best_r)
            print("Start training...")
            l.train(context[arm_select], r, local_training_iter)

        print("iter {0} --- reward: {1}".format(t, r))
        print(f"Best value found till now: {best_r}")
        best_values.append(best_r)

    print('Evaluate on test data...')
    prompts = model_forward_api.return_best_prompt()
    print("Best instruction is:")
    print(prompts)

    prompts_set = model_forward_api.return_prompts_set()
    print("The final instruction set is:")
    print(prompts_set)
    prompts_list = model_forward_api.return_prompts_list()


    # Evaluate on test data
    print('Evaluating on test data...')

    test_conf = {
        'generation': {
            'num_subsamples': 3,
            'num_demos': 5,
            'num_prompts_per_subsample': 0,
            'model': {
                'gpt_config': {
                    'model': gpt
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator, # option: accuracy (cannot use likelihood here due to the textual outputs from ChatGPT do not have log prob)
            'task': task,
            'num_samples': min(100, len(test_data[0])),
            'model': {
                "name": "GPT_forward",
                'gpt_config': {
                   'model': gpt
                }
            }
        }
    }
    
    test_res = ape.evaluate_prompts(prompts=prompts,
                                    eval_template=eval_template,
                                    eval_data=test_data,
                                    few_shot_data=prompt_gen_data,
                                    demos_template=demos_template,
                                    conf=test_conf,
                                    base_conf=base_conf)
    test_res = test_res[0]
    test_score = test_res.sorted()[1][0]
    return test_score, prompts, prompts_list, best_values
    # print(f'Test score on ChatGPT: {test_score}')

def parse_args():
    parser = argparse.ArgumentParser(description="InstructZero pipeline")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--n_prompt_tokens",
        type=int,
        default=5,
        help="The number of prompt tokens."
    )
    parser.add_argument(
        "--HF_cache_dir",
        type=str,
        default="PATH",
        help="Your vicuna directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed."    
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.1,
        help="Set the parameter nu."    
    )
    parser.add_argument(
        "--lamdba",
        type=float,
        default=0.1,
        help="Set the lamdba parameter."    
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=40,
        help="Set the number of initialization points."    
    )
    parser.add_argument(
        "--n_domain",
        type=int,
        default=10000,
        help="Set the number of domain."    
    )
    parser.add_argument(
        "--total_iter",
        type=int,
        default=165,
        help="Set the number of total queries."    
    )
    parser.add_argument(
        "--local_training_iter",
        type=int,
        default=30,
        help="Set the number of total queries."    
    )
    parser.add_argument(
        "--random_proj",
        type=str,
        default='uniform',
        help="Set the projection method."    
    )
    parser.add_argument(
        "--intrinsic_dim",
        type=int,
        default=100,
        help="Set the number of intrinsic dim."    
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=1000,
        help="Set the number of domains to be evaluated at each ucb iteration."    
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Set the name of the experiments."    
    )
    parser.add_argument(
        "--gpt",
        type=str,
        default="gpt-3.5-turbo",
        help="Which version of gpt to use."    
    )
    parser.add_argument(
        "--init_scale",
        type=float,
        default=1,
        help="Which scale to use."    
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="last",
        help="Which pooling method to use."    
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(set_all_seed(args.seed))
    test_score, prompts, prompts_list, best_values = run(
        task=args.task,
        n_prompt_tokens=args.n_prompt_tokens,
        HF_cache_dir=args.HF_cache_dir,
        nu=args.nu,
        lamdba=args.lamdba,
        n_init=args.n_init,
        n_domain=args.n_domain,
        total_iter=args.total_iter,
        local_training_iter = args.local_training_iter,
        random_proj=args.random_proj,
        intrinsic_dim=args.intrinsic_dim,
        n_eval=args.n_eval,
        gpt=args.gpt,
        init_scale=args.init_scale,
        pooling=args.pooling
    )
    
    args_dict = vars(args)
    args_dict['test_score'] = test_score
    args_dict['best_prompt'] = prompts
    args_dict['prompts_list'] = prompts_list
    args_dict['best_values'] = best_values

    save_dir = "./NeuralBanditResult0301"
    # get a path with the current time
    path = os.path.join(save_dir,args.task + datetime.datetime.now().strftime("-%Y-%m-%d_%H-%M-%S")+ args.name +".json")

    with open(path, 'x') as fp:
        json.dump(args_dict, fp, indent=4)
    
    print("Finished!!!")
    print(f'Test score on ChatGPT: {test_score}')


