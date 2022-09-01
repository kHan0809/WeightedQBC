import torch
import torch.nn as nn
from Model.model import Qnet, Policy, soft_update, hard_update
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import deque
from copy import deepcopy
import torch.nn.functional as F

class CustomDataSet(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

class Buffer:
    def __init__(self,o_dim,a_dim,buffer_size = 1000000):
        self.size = buffer_size
        self.num_experience = 0
        self.o_mem = np.empty((self.size, o_dim), dtype=np.float32)
        self.a_mem = np.empty((self.size, a_dim), dtype=np.float32)
        self.no_mem = np.empty((self.size, o_dim), dtype=np.float32)
        self.r_mem = np.empty((self.size, 1), dtype=np.float32)
        self.done_mem = np.empty((self.size, 1), dtype=np.float32)
    def store_sample(self,o,a,r,no,done):
        idx = self.num_experience%self.size
        self.o_mem[idx] = o
        self.a_mem[idx] = a
        self.r_mem[idx] = r
        self.no_mem[idx] = no
        self.done_mem[idx] = done
        self.num_experience += 1
    def random_batch(self, batch_size = 256):
        N = min(self.num_experience, self.size)
        idx = np.random.choice(N,batch_size)
        o_batch = self.o_mem[idx]
        a_batch = self.a_mem[idx]
        r_batch = self.r_mem[idx]
        no_batch = self.no_mem[idx]
        done_batch = self.done_mem[idx]
        return o_batch, a_batch, r_batch, no_batch, done_batch
    def all_batch(self):
        N = min(self.num_experience,self.size)
        return self.o_mem[:N], self.a_mem[:N], self.r_mem[:N], self.no_mem[:N], self.done_mem[:N]
    def store_demo(self,demo):
        demo_len= len(demo)-1
        self.o_mem[:demo_len]  = demo[:-1]
        self.no_mem[:demo_len] = demo[1:]
        self.num_experience += demo_len



class QBC_agent:
    def __init__(self,o_dim,a_dim,args):
        self.o_dim, self.a_dim = o_dim, a_dim
        self.args = args
        #SAC Hyperparameters
        self.gamma = args.gamma
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.lr = args.lr
        self.q_update_count = 0
        self.n_actions = 50

        #Define networks
        self.q1 = Qnet(self.o_dim, self.a_dim, self.hidden_size).to(args.device_train)
        self.q2 = Qnet(self.o_dim, self.a_dim, self.hidden_size).to(args.device_train)
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)

        self.pi = Policy(self.o_dim, self.a_dim,self.hidden_size).to(args.device_train)
        self.target_pi = deepcopy(self.pi)
        self.buffer = Buffer(o_dim, a_dim)

        #Define optimizer
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.lr)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=self.lr)
        #====cql hyper====
        # CQL Hyperparmeters==--=나중에 args로 바꿔줘야함
        # self.backup_entropy = False # 보류
        self.cql_n_actions = 10
        self.cql_importance_sample = True
        self.cql_lagrange = False
        self.cql_target_action_gap = 1.0
        self.cql_temp = 1.0
        self.cql_min_q_weight = 5.0
        self.cql_max_target_backup = False
        self.cql_clip_diff_min = -np.inf
        self.cql_clip_diff_max =  np.inf

    def init_pi(self,task_name,iter,path):
        path = path + '/' + task_name + '-' + str(iter) + '.pt'
        self.pi.load_state_dict(torch.load(path)['policy'])
        self.target_pi = deepcopy(self.pi)
    def init_q(self,task_name,iter,path):
        path = path + '/' + task_name + '-' + str(iter) + '.pt'
        self.q1.load_state_dict(torch.load(path)['q1'])
        self.q2.load_state_dict(torch.load(path)['q2'])
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)

    def save_checkpoint(self,task_name,iter,path):
        path = path +'/'+task_name+'-'+ str(iter) + '.pt'
        print('Saving model to {}'.format(path))
        torch.save({'policy': self.pi.state_dict(),
                    'q1'    : self.q1.state_dict(),
                    'q2'    : self.q2.state_dict(),
                    }, path)

    def select_action(self,o,eval=False):
        o = o.reshape([1,-1])
        if eval:
            action = self.pi(torch.FloatTensor(o).to(self.args.device_train))
            return  action.cpu().detach().numpy()[0]
        else:
            action = (self.pi(torch.FloatTensor(o).to(self.args.device_train)).cpu().detach().numpy()[0] + \
                     + 0.1 * np.random.normal(0.0, 1.0, [self.a_dim])).clip(-1.0,1.0)
            return action

    def store_sample(self,o,a,r,no,done):
        self.buffer.store_sample(o,a,r,no,done)

    def train_bc(self,batch):
        self.pi.train()
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)

        self.pi_opt.zero_grad()
        pred_action = self.pi(state_batch)
        action_loss = F.mse_loss(pred_action,action_batch)
        action_loss.backward()
        self.pi_opt.step()

    def train_Q(self, batch, cql=False):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)
        reward_batch = torch.FloatTensor(reward_batch).to(self.args.device_train)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.args.device_train)
        done_batch = torch.FloatTensor(done_batch).to(self.args.device_train)

        if cql:
            self.q_train_cql(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        else:
            self.q_train(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

        self.q_update_count += 1
        # if self.q_update_count%2 == 0:
        with torch.no_grad():
            soft_update(self.target_q1, self.q1, self.tau)
            soft_update(self.target_q2, self.q2, self.tau)

    def train_QBC(self,batch):

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)

        self.pi.train()
        random_actions = action_batch.new_empty((state_batch.shape[0], self.n_actions, self.a_dim),requires_grad=False).uniform_(-1, 1)
        with torch.no_grad():
            q_random_val1,q_random_val2 = self.q1(state_batch,random_actions), self.q2(state_batch,random_actions) #state는 그대로 [256,17] action은 [256,10,6이 들어가면된다.]
            min_q = torch.min(q_random_val1, q_random_val2).mean(dim=1)

            q_val1, q_val2 = self.q1(state_batch, action_batch), self.q2(state_batch, action_batch)

            weight = torch.min((q_val1 - min_q) / torch.abs(min_q), (q_val2 - min_q) / torch.abs(min_q))
            weight = weight.clamp(0.0, 2.0)

        self.pi_opt.zero_grad()
        pred_action = self.pi(state_batch)
        action_loss = torch.mean(((pred_action - action_batch)**2)*weight.reshape(-1, 1))
        action_loss.backward()
        self.pi_opt.step()


    def test_q(self,batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)
        with torch.no_grad():
            q_val1, q_val2 = self.q1(state_batch, action_batch), self.q2(state_batch, action_batch)
        return q_val1, q_val2

    def q_train(self,state_batch,action_batch,reward_batch,next_state_batch,done_batch):
        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        q_val1,q_val2 = self.q1(state_batch,action_batch), self.q2(state_batch,action_batch)
        # reward_batch, done_batch = reward_batch.reshape(q_val1.shape[0]), done_batch.reshape(q_val1.shape[0])
        with torch.no_grad():
            noise = (torch.randn_like(action_batch) * 0.2).clamp(-0.5, 0.5)
            next_action_batch = (self.target_pi(next_state_batch) + noise).clamp(-1.,1.)
            next_q_val1, next_q_val2 = self.target_q1(next_state_batch,next_action_batch), self.target_q2(next_state_batch,next_action_batch)
            minq = torch.min(next_q_val1,next_q_val2)
            target_ = reward_batch + self.gamma*(1-done_batch)*minq

        q1_loss = F.mse_loss(target_,q_val1)
        q2_loss = F.mse_loss(target_,q_val2)

        q_loss = q1_loss + q2_loss
        q_loss.backward()
        self.q1_opt.step()
        self.q2_opt.step()

    def q_train_cql(self,state_batch,action_batch,reward_batch,next_state_batch,done_batch):
        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        q_val1,q_val2 = self.q1(state_batch,action_batch), self.q2(state_batch,action_batch)
        with torch.no_grad():
            noise = (torch.randn_like(action_batch) * 0.2).clamp(-0.5, 0.5)
            next_action_batch = (self.target_pi(next_state_batch) + noise).clamp(-1.,1.)
            next_q_val1, next_q_val2 = self.target_q1(next_state_batch,next_action_batch), self.target_q2(next_state_batch,next_action_batch)
            minq = torch.min(next_q_val1,next_q_val2)
            target_ = reward_batch + self.gamma*(1-done_batch)*minq

        q1_loss = F.mse_loss(target_,q_val1)
        q2_loss = F.mse_loss(target_,q_val2)


        #====여까지는 그냥 SAC랑 같음
        batch_size = action_batch.shape[0]
        action_dim = action_batch.shape[-1]
        cql_random_actions = action_batch.new_empty((batch_size, self.cql_n_actions, action_dim),requires_grad=False).uniform_(-1, 1)


        cql_current_actions = self.pi(state_batch, repeat=self.cql_n_actions)
        cql_next_actions    = self.pi(next_state_batch, repeat=self.cql_n_actions)
        cql_current_actions, cql_next_actions = cql_current_actions.detach(), cql_next_actions.detach()


        cql_q1_rand = self.q1(state_batch, cql_random_actions)
        cql_q2_rand = self.q2(state_batch, cql_random_actions)
        cql_q1_current_actions = self.q1(state_batch, cql_current_actions)
        cql_q2_current_actions = self.q2(state_batch, cql_current_actions)
        cql_q1_next_actions = self.q1(state_batch, cql_next_actions)
        cql_q2_next_actions = self.q2(state_batch , cql_next_actions)

        cql_cat_q1 = torch.cat(
            [cql_q1_rand, torch.unsqueeze(q_val1, 1), cql_q1_next_actions, cql_q1_current_actions], dim=1
        )
        cql_cat_q2 = torch.cat(
            [cql_q2_rand, torch.unsqueeze(q_val2, 1), cql_q2_next_actions, cql_q2_current_actions], dim=1
        )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q_val1,
            -np.inf,
             np.inf,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q_val2,
            -np.inf,
             np.inf,
        ).mean()

        cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
        cql_min_qf2_loss = cql_qf2_diff * self.cql_min_q_weight

        q1_loss_add_cql =  q1_loss + cql_min_qf1_loss
        q2_loss_add_cql =  q2_loss + cql_min_qf2_loss


        q1_loss_add_cql.backward(retain_graph=True)
        self.q1_opt.step()
        q2_loss_add_cql.backward()
        self.q2_opt.step()







