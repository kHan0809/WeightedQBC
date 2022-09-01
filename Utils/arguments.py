import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default="halfcheetah-random-v2", help="halfcheetah-medium-expert-v2 halfcheetah-medium-v2 halfcheetah-expert-v2")

    parser.add_argument('--device_eval',  default="cpu")
    parser.add_argument('--device_train', default="cuda")

    # ===================hyperparameter======================
    parser.add_argument('--gamma', type=float, default=0.99) #TD3 공용
    parser.add_argument('--tau', type=float, default=0.005)  #TD3 공용
    parser.add_argument('--lr', type=float, default=3e-4, help="3e-4")
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)

    # ====================train===============
    parser.add_argument('--train_num_per_epoch', type=int, default=1000)
    parser.add_argument('--bc_train_epoch',      type=int, default=50)
    parser.add_argument('--q_train_epoch', type=int, default=50)
    parser.add_argument('--q_idx', type=int, default=0)
    parser.add_argument('--cql', type=bool,  default=True)
    parser.add_argument('--qbc_train_epoch', type=int, default=1000)

    #=====================eval===============
    parser.add_argument('--eval_num',    type=int, default=5)
    parser.add_argument('--eval_period', type=int, default=5)




    args = parser.parse_args()
    return args

