from transformers import LlamaForSequenceClassification, LlamaModel, LlamaPreTrainedModel, LlamaForCausalLM
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from backpack import backpack, extend
from backpack.extensions import BatchGrad
# import torch
# import torch.nn as nn
# from functorch import make_functional, vmap, vjp, jvp, jacrev

tkwargs = {
    "device": torch.device("cuda:0"),
    "dtype": torch.double,
}

class MLPRegression(nn.Module):

    def __init__(self, input_dim=86):
        super(MLPRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

class CustomImageDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]

# fnet, params = make_functional(ini_model)

# def empirical_ntk_jacobian_contraction(fnet, params, x1 , batch_size=128, cpu=False, regression=False, return_features=False):
#     def fnet_single(params, x):
#         if regression:
#             return fnet(params, x.unsqueeze(0)).squeeze(0)
#         else:
#             # taking only the first element to compute jacobian
#             return fnet(params, x.unsqueeze(0)).squeeze(0)[0]
    

#     def get_jac(x, cpu=False):
#         def get_batch_jac(x_batch):
#             jac = vmap(jacrev(fnet_single), (None, 0))(params, x_batch)
#             jac = [j.flatten(1) for j in jac]
#             jac = torch.hstack(jac)
#             return jac
#         num_dp = x.shape[0]
#         batch_num = num_dp // batch_size
#         residule = num_dp % batch_size
#         all_jac = []
#         en=0
#         for idx in range(batch_num):
#             st = idx * batch_size
#             en = (idx + 1) * batch_size
#             jac = get_batch_jac(x[st:en])
#             if cpu:
#                 jac = jac.cpu()
#             all_jac.append(jac)
#         if residule:
#             jac = get_batch_jac(x[en:])
#             if cpu:
#                 jac = jac.cpu()
#             all_jac.append(jac)
#         all_jac = torch.vstack(all_jac)
#         return all_jac
    
#     # Compute J(x1)
#     same_data =  x1.data_ptr() ==  x2.data_ptr() and x1.shape[0] == x2.shape[0] # the same data
#     if same_data:
#         all_jac = get_jac(x1, cpu=cpu)
#         result = all_jac @ all_jac.transpose(0,1)
#         if not return_features:
#             del all_jac
#     else:
#         all_jac1 = get_jac(x1, cpu=cpu)
#         all_jac2 = get_jac(x2, cpu=cpu)
#         result = all_jac1 @ all_jac2.transpose(0,1)
#         del all_jac1, all_jac2
#     if return_features:
#         return result.cpu().detach().numpy(), all_jac.cpu().detach().numpy()
#     else:
#         return result.cpu().detach().numpy()
    

class Network(nn.Module):
    def __init__(self, input_dim, hidden_size=100, depth=1, init_params=None):
        super(Network, self).__init__()

        self.activate = nn.ReLU()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, hidden_size))
        for i in range(depth-1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))
        
        if init_params is None:
            ## use initialization
            for i in range(len(self.layer_list)):
                torch.nn.init.normal_(self.layer_list[i].weight, mean=0, std=1.0)
                torch.nn.init.normal_(self.layer_list[i].bias, mean=0, std=1.0)
        else:
            ### manually set the initialization vector
            for i in range(len(self.layer_list)):
                self.layer_list[i].weight.data = init_params[i*2]
                self.layer_list[i].bias.data = init_params[i*2+1]
    
    def forward(self, x):
        y = x
        for i in range(len(self.layer_list)-1):
            y = self.activate(self.layer_list[i](y))
        y = self.layer_list[-1](y)
        return y


class NeuralTSDiag:
    def __init__(self, input_dim, lamdba=1, nu=1, style='ucb', init_x=None, init_y=None, diagonalize=True):

        self.diagonalize = diagonalize
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.func = extend(Network(input_dim).to(**tkwargs))
        self.init_state_dict = deepcopy(self.func.state_dict())

        if init_x is not None:
            self.context_list = init_x.to(dtype=torch.float32)
        else:
            self.context_list = None
        if init_y is not None:
            self.reward = init_y.to(dtype=torch.float32)
        else:
            self.reward = None
        self.len = 0
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)

        if self.diagonalize:
            ### diagonalization
            self.U = lamdba * torch.ones((self.total_param,))
        else:
            ### no diagonalization
            self.U = lamdba * torch.diag(torch.ones((self.total_param,)))
        
        self.nu = nu
        self.style = style
        self.loss_func = nn.MSELoss()
        self.mean = None
        self.std = None


    def select(self, context, batch_size=300):     
        if self.mean is not None:
            context_ = (context - self.mean) / self.std   
        else:
            context_ = context
        # batch computing of jacobian
        # batch_size = 300
        context_size = context_.shape[0]        
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        g_list = []
        mu = []
        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context_[(i*batch_size):]
            else:
                context_batch = context_[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch)
            sum_mu = torch.sum(mu_)
            with backpack(BatchGrad()):
                sum_mu.backward()                
            g_list_ = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)
            g_list.append(g_list_.cpu())
            mu.append(mu_.cpu())
        g_list = torch.vstack(g_list)
        mu = torch.vstack(mu)
        # mu = self.func(context).cpu()
        # sum_mu = torch.sum(mu)
        # with backpack(BatchGrad()):
        #     sum_mu.backward()

        # g_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1).cpu()

        if self.diagonalize:
#             ### diagonalization
            sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * g_list * g_list / self.U, dim=1))
        else:
            ### no diagonalization
            tmp = torch.matmul(g_list, torch.inverse(self.U))
            sigma = torch.sqrt(self.nu * self.lamdba * torch.matmul(tmp, torch.transpose(g_list, 0, 1)))
            sigma = torch.diagonal(sigma, 0)

        if self.style == 'ts':
            sample_r = torch.normal(mu.view(-1), sigma.view(-1))
        elif self.style == 'ucb':
            sample_r = mu.view(-1) + sigma.view(-1)
        arm = torch.argmax(sample_r)

        if self.diagonalize:
            ### diagonalization
            self.U += g_list[arm] * g_list[arm]
        else:
            ### no diagonalization
            self.U += torch.outer(g_list[arm], g_list[arm])

        return arm, g_list[arm].norm().item()


    def train(self, context, reward, local_training_iter=30):
        if self.init_state_dict is not None:
            self.func.load_state_dict(deepcopy(self.init_state_dict))
        if context is not None:
            if self.context_list is None:
                self.context_list = torch.from_numpy(context.reshape(1, -1)).to(**tkwargs)
                self.reward = torch.tensor([reward]).to(**tkwargs)
            else:
                self.context_list = torch.cat((self.context_list, context.reshape(1, -1).to(**tkwargs)))
                self.reward = torch.cat((self.reward, torch.tensor([reward]).reshape(1,-1).to(**tkwargs)))

        self.len = self.context_list.shape[0]
        optimizer = torch.optim.Adam(self.func.parameters(), lr=1e-3, weight_decay=self.lamdba / self.len)

        # if self.len % self.delay != 0:
        #     return 0
        # torch.save({"context_list": self.context_list, "reward": self.reward}, 'train_data.pt')

        self.std = self.context_list.std(dim=0) + 1e-30
        self.mean = self.context_list.mean(dim=0)
        standardized_context = (self.context_list - self.mean) / self.std 
        # standardized_reward = ((self.reward - self.reward.mean(dim=0)) / (self.reward.std(dim=0) + 1e-30)).reshape(-1)
        standardized_reward = self.reward.reshape(-1)
        for _ in range(local_training_iter):
            self.func.zero_grad()
            optimizer.zero_grad()
            pred = self.func(standardized_context).view(-1)
        
            loss = self.loss_func(pred, standardized_reward)
            loss.backward()
            optimizer.step()
        print("Training Loss : ", loss.item())
        return self.func.state_dict()


class NeuralTSDiag_Save:
    def __init__(self, input_dim, lamdba=1, nu=1, style='ucb', init_x=None, init_y=None, diagonalize=True):

        self.diagonalize = diagonalize
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.func = extend(Network(input_dim).to(**tkwargs))
        self.init_state_dict = deepcopy(self.func.state_dict())

        if init_x is not None:
            self.context_list = init_x.to(dtype=torch.float32)
        else:
            self.context_list = None
        if init_y is not None:
            self.reward = init_y.to(dtype=torch.float32)
        else:
            self.reward = None
        self.len = 0
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)

        if self.diagonalize:
            ### diagonalization
            self.U = lamdba * torch.ones((self.total_param,))
        else:
            ### no diagonalization
            self.U = lamdba * torch.diag(torch.ones((self.total_param,)))
        
        self.nu = nu
        self.style = style
        self.loss_func = nn.MSELoss()
        self.mean = None
        self.std = None


    def select(self, context, batch_size=300):     
        if self.mean is not None:
            context_ = (context - self.mean) / self.std   
        else:
            context_ = context
        # batch computing of jacobian
        # batch_size = 300
        context_size = context_.shape[0]        
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        mu = []
        sigma = []
        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context_[(i*batch_size):]
            else:
                context_batch = context_[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch)
            sum_mu = torch.sum(mu_)
            with backpack(BatchGrad()):
                sum_mu.backward()                
            g_list_ = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1).cpu()
            mu.append(mu_.cpu())

            if self.diagonalize:
                ### diagonalization
                sigma_ = torch.sqrt(torch.sum(self.lamdba * self.nu * g_list_ * g_list_ / self.U, dim=1))
            else:
                ### no diagonalization
                tmp = torch.matmul(g_list_, torch.inverse(self.U))
                sigma_ = torch.sqrt(self.nu * self.lamdba * torch.matmul(tmp, torch.transpose(g_list_, 0, 1)))
                sigma_ = torch.diagonal(sigma_, 0)
            sigma.append(sigma_)

        mu = torch.vstack(mu)
        sigma = torch.hstack(sigma)
        # mu = self.func(context).cpu()
        # sum_mu = torch.sum(mu)
        # with backpack(BatchGrad()):
        #     sum_mu.backward()

        # g_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1).cpu()

        if self.style == 'ts':
            sample_r = torch.normal(mu.view(-1), sigma.view(-1))
        elif self.style == 'ucb':
            sample_r = mu.view(-1) + sigma.view(-1)
        arm = torch.argmax(sample_r)

        mu_ = self.func(context_[[arm.item(), arm.item()]])
        sum_mu = torch.sum(mu_)
        with backpack(BatchGrad()):
            sum_mu.backward()                
        g_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)[0].reshape(-1).cpu()
        
        if self.diagonalize:
            ### diagonalization
            self.U += g_list * g_list
        else:
            ### no diagonalization
            self.U += torch.outer(g_list, g_list)

        return arm, g_list.norm().item()


    def train(self, context, reward, local_training_iter=30):
        if self.init_state_dict is not None:
            self.func.load_state_dict(deepcopy(self.init_state_dict))
        if context is not None:
            if self.context_list is None:
                self.context_list = torch.from_numpy(context.reshape(1, -1)).to(**tkwargs)
                self.reward = torch.tensor([reward]).to(**tkwargs)
            else:
                self.context_list = torch.cat((self.context_list, context.reshape(1, -1).to(**tkwargs)))
                self.reward = torch.cat((self.reward, torch.tensor([reward]).reshape(1,-1).to(**tkwargs)))

        self.len = self.context_list.shape[0]
        optimizer = torch.optim.Adam(self.func.parameters(), lr=1e-3, weight_decay=self.lamdba / self.len)

        # if self.len % self.delay != 0:
        #     return 0
        # torch.save({"context_list": self.context_list, "reward": self.reward}, 'train_data.pt')

        self.std = self.context_list.std(dim=0) + 1e-30
        self.mean = self.context_list.mean(dim=0)
        standardized_context = (self.context_list - self.mean) / self.std 
        # standardized_reward = ((self.reward - self.reward.mean(dim=0)) / (self.reward.std(dim=0) + 1e-30)).reshape(-1)
        standardized_reward = self.reward.reshape(-1)
        for _ in range(local_training_iter):
            self.func.zero_grad()
            optimizer.zero_grad()
            pred = self.func(standardized_context).view(-1)
        
            loss = self.loss_func(pred, standardized_reward)
            loss.backward()
            optimizer.step()
        print("Training Loss : ", loss.item())
        return self.func.state_dict()


class NeuralTS:
    def __init__(self, input_dim, lamdba=1, style='ucb', init_x=None, init_y=None, diagonalize=True, lr=0.01, beta=1, opt="SGD", local_training_iter=1000):

        self.diagonalize = diagonalize
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.func = Network(input_dim).to(**tkwargs)
        self.func_1 = extend(Network(input_dim).to(**tkwargs))
        self.func_2 = Network(input_dim).to(**tkwargs)
        # self.init_state_dict = deepcopy(self.func.state_dict())

        if init_x is not None:
            self.context_list = init_x.to(dtype=torch.double)
        else:
            self.context_list = None
        if init_y is not None:
            self.reward = init_y.to(dtype=torch.double)
        else:
            self.reward = None
        self.len = 0
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)

        if self.diagonalize:
            ### diagonalization
            self.U = lamdba * torch.ones((self.total_param,))
        else:
            ### no diagonalization
            self.U = lamdba * torch.diag(torch.ones((self.total_param,)))
        
        self.style = style
        self.loss_func = nn.MSELoss()
        self.lr = lr
        self.beta = beta
        self.local_training_iter = local_training_iter
        self.opt = opt


    def select(self, context):      

        def weights_init(m, W_std=1.5, b_std=0.05):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=W_std)

                # You can also include bias if the layer has it
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0., std=b_std)
        def reset_last_layer(m):
            if isinstance(m, nn.Linear):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()

        self.func_1.apply(lambda m: weights_init(m, W_std=1.5, b_std=0.05))
        self.func.load_state_dict(deepcopy(self.func_1.state_dict()))
        self.func_2.apply(lambda m: weights_init(m, W_std=1.5, b_std=0.05))

        # Get the last layer in the model
        last_layer = list(self.func_2.modules())[-1]
        reset_last_layer(last_layer)
        self.w_list = torch.cat([p.data.view(1,-1).detach() for p in self.func_2.parameters()], dim=1).cpu()

        self.retrain()
        
        # batch computing of jacobian
        batch_size = 300
        context_size = context.shape[0]        
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        g_list = []
        mu = []
        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context[(i*batch_size):]
            else:
                context_batch = context[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func_1(context_batch)
            sum_mu = torch.sum(mu_)
            with backpack(BatchGrad()):
                sum_mu.backward()                
            g_list_ = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func_1.parameters()], dim=1)
            g_list.append(g_list_.cpu())
            mu_ori = self.func(context_batch)
            mu.append(mu_ori.cpu())
        g_list = torch.vstack(g_list)
        mu = torch.vstack(mu)

        delta_value = g_list @ self.w_list.view(-1,1)
        
        sample_r = mu + delta_value
        
        arm = torch.argmax(sample_r)

        return arm, g_list[arm].norm().item()


    def retrain(self):
        self.len = self.context_list.shape[0]
        optimizer = eval("torch.optim." + self.opt)(self.func.parameters(), lr=self.lr, weight_decay=self.lamdba / self.len)
        
        # compute the point wise gradient    
        mu_ = self.func_1(self.context_list)

        sum_mu = torch.sum(mu_)
        with backpack(BatchGrad()):
            sum_mu.backward()                
        g_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func_1.parameters()], dim=1)
        g_list = g_list.cpu()
        
        delta_value = g_list @ self.w_list.view(-1,1)
        
        for _ in range(self.local_training_iter ):
            self.func.zero_grad()
            optimizer.zero_grad()
            pred = self.func(self.context_list).view(-1) * self.beta + delta_value.view(-1).cuda()
        
            loss = self.loss_func(pred, self.reward.reshape(-1))
            loss.backward()
            optimizer.step()
        print("Training Loss : ", loss.item())
        return self.func.state_dict()

    def add_data(self, context, reward):
        size = context.shape[0]
        if context is not None:
            if self.context_list is None:
                self.context_list = torch.from_numpy(context.reshape(1, -1)).to(**tkwargs)
                self.reward = torch.tensor([reward]).to(**tkwargs)
            else:
                # print(self.context_list.shape)
                # print(context.reshape(size, -1).shape)
                # raise ValueError
                self.context_list = torch.cat((self.context_list, context.reshape(size, -1).to(**tkwargs)))
                self.reward = torch.cat((self.reward, torch.tensor([reward]).reshape(size,-1).to(**tkwargs)))

class MLPRegression_Train:
    def __init__(
        self,
        input_dim=4096,
        optimizer_fn=torch.optim.Adam,
        loss_fn=nn.MSELoss,
        lr=0.001,
        batch_size=64,
        epochs=30,
        device=None):

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.model = MLPRegression(input_dim).to(device)
        self.optimizer = optimizer_fn(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn()
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        # backup for the initial model weight and optimizer
        self.init_model_weight = deepcopy(self.model.state_dict())
        self.optimizer_fn = optimizer_fn
    
    def get_data_loader(self, X_train, Y_train):
        dataset = CustomImageDataset(X_train, Y_train)
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return train_dataloader
        
    def fit(self, X_train, Y_train, verbose=False, epochs=None):
        if epochs == None:
            epochs = self.epochs

        train_loader = self.get_data_loader(X_train, Y_train)
        for e in range(epochs):
            self.model.train()
            
            # running local epochs
            for batch_idx, batch in enumerate(train_loader):
                data, label = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(data)
                loss = self.loss_fn(pred, label)
                loss.backward()
                self.optimizer.step()
            
            if verbose:
                print('Epoch: {}, Loss: {:.4f}'.format(e, loss))

        return self.model

    def select(self, context, diagonalize, lamdba, nu, style, ):
        self.model.train()
        mu = self.model(context)
        sum_mu = torch.sum(mu)
        with backpack(BatchGrad()):
            sum_mu.backward()

        g_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)

        if diagonalize:
#             ### diagonalization
            sigma = torch.sqrt(torch.sum(lamdba * nu * g_list * g_list / self.U, dim=1))
        else:
            ### no diagonalization
            tmp = torch.matmul(g_list, torch.inverse(self.U))
            sigma = torch.sqrt(nu * lamdba * torch.matmul(tmp, torch.transpose(g_list, 0, 1)))
            sigma = torch.diagonal(sigma, 0)

        if style == 'ts':
            sample_r = torch.normal(mu.view(-1), sigma.view(-1))
        elif style == 'ucb':
            sample_r = mu.view(-1) + sigma.view(-1)
        arm = torch.argmax(sample_r)

        if diagonalize:
            ### diagonalization
            self.U += g_list[arm] * g_list[arm]
        else:
            ### no diagonalization
            self.U += torch.outer(g_list[arm], g_list[arm])

        return arm, g_list[arm].norm().item()

    def restart_model(self):
        self.model.load_state_dict(deepcopy(self.init_model_weight))
        self.optimizer = self.optimizer_fn(self.model.parameters(), lr=self.lr)


class LlamaForMLPRegression(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def get_last_token_hidden_state(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sequence_lengths: Optional[int] = None,
        n_prompt_tokens: Optional[int] = 0,
        pooling: Optional[str] = "last",
    ) -> Tuple:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if sequence_lengths is None:
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1 + n_prompt_tokens).to(hidden_states.device)
                else:
                    sequence_lengths = -1
        if pooling == "last":
            pooled_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        elif pooling == "mean":
            pooled_states = hidden_states.mean(dim=1)
        elif pooling == "max":
            pooled_states = hidden_states.max(dim=1).values
        return (pooled_states,)
