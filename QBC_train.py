import copy

import gym
from Model.class_model import  QBC_agent
from Utils.arguments import get_args
import d4rl
from Utils.utils import d4rl_dataset, Eval


args = get_args()

# env = gym.make("InvertedPendulum-v2")
env = gym.make(args.task_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]
epi_length = env.spec.max_episode_steps


agent = QBC_agent(state_dim,action_dim,args)
dataset = d4rl_dataset(env.unwrapped)
epoch_count, max_return, max_idx = 0, -100, 0
#==========BC train====================
for num in range(args.bc_train_epoch):
  for step in range(args.train_num_per_epoch):
    batch = dataset.get_data()
    agent.train_bc(batch)
  epoch_count += 1

  #====Eval====
  if epoch_count % args.eval_period == 0:
    mean_return = Eval(env,agent,epoch_count,args)
    if mean_return > max_return:
      max_return, max_idx = copy.deepcopy(mean_return), copy.deepcopy(epoch_count)
      agent.save_checkpoint(args.task_name+str(args.cql), epoch_count, "./model_save/bc")
print("[BC MAX_IDX] : ", max_idx)

#=============Q train=====================
epoch_count = 0
agent.init_pi(args.task_name+str(args.cql), max_idx, "./model_save/bc")
for num in range(args.q_train_epoch):
  for step in range(args.train_num_per_epoch):
    batch = dataset.get_data()
    agent.train_Q(batch,args.cql)

  epoch_count += 1
  if epoch_count % args.eval_period == 0:
    q1, q2 = agent.test_q(batch)
    print("[epoch] :", epoch_count, "Q1 : ", sum(q1) / batch[0].shape[0], "Q2 : ", sum(q2) / batch[0].shape[0])
    agent.save_checkpoint(args.task_name+str(args.cql), args.q_idx, "./model_save/bc_q")

#==========QBC train====================
epoch_count, max_return, max_idx = 0, 0, 0
agent.init_pi(args.task_name+str(args.cql), args.q_idx, "./model_save/bc_q")
agent.init_q(args.task_name+str(args.cql),  args.q_idx, "./model_save/bc_q")
for num in range(args.qbc_train_epoch):
  for step in range(args.train_num_per_epoch):
    batch = dataset.get_data()
    agent.train_QBC(batch)
  epoch_count += 1

  #====Eval====
  if epoch_count % args.eval_period == 0:
    mean_return = Eval(env,agent,epoch_count,args)
    if mean_return > max_return:
      max_return, max_idx = copy.deepcopy(mean_return), copy.deepcopy(epoch_count)
      agent.save_checkpoint(args.task_name+str(args.cql), epoch_count, "./model_save/qbc")
print("[QBC MAX_IDX] : ", max_idx)
