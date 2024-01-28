# time: 2024/1/27 16:18
# author: YanJP
import envs
import para
from MAPPO_MPE_main import *




class Baseline1():   # 随机算法
    def __init__(self,BSs):
        self.rewards=[]
        self.BSs=BSs
        for b in self.BSs:
            b.video_cache=b.ini_state[0]
            b.all_video_hot=b.ini_state[1]
        pass
    def deal_each_bs(self, bs: envs.Basestation):
        prob = bs.all_video_hot
        random_request = np.random.choice(np.arange(para.num_videos), size=para.request_times, p=prob)
        rs = []
        for v in random_request:
            if v in bs.video_cache:
                r = 1
            elif v in self.all_cache:
                r = 1 - para.delay_n
            else:
                r = 1 - para.delay_Cloud
            rs.append(r)
        return sum(rs)
    def get_QoE(self,):
        for _ in range(para.Times):
            self.all_cache=np.concatenate((self.BSs[0].video_cache,self.BSs[1].video_cache,self.BSs[2].video_cache))
            self.all_cache=np.unique(self.all_cache)
            r=0
            for bs in self.BSs:
                reward=self.deal_each_bs(bs)
                r+=reward
            self.rewards.append(r)
        return sum(self.rewards)
    def get_avg_Qoe(self,):
        res=[]
        for i in range(para.test_times):
            self.rewards=[]
            r=self.get_QoE()
            res.append(r)
        print("Baseline1:",sum(res)/len(res))
        return sum(res)/len(res)
class Baseline2():   # 非协作做法
    def __init__(self,BSs):
        self.rewards=[]
        self.BSs=BSs
        for b in self.BSs:
            b.video_cache=b.ini_state[0]
            b.all_video_hot=b.ini_state[1]
    def deal_each_bs(self, bs: envs.Basestation):
        prob = bs.all_video_hot
        random_request = np.random.choice(np.arange(para.num_videos), size=para.request_times, p=prob)
        rs = []
        for v in random_request:
            if v in bs.video_cache:
                r = 1
            else:
                r = 1 - para.delay_Cloud
            rs.append(r)
        return sum(rs)
    def get_QoE(self,):
        for _ in range(para.Times):
            # self.all_cache=np.concatenate((self.BSs[0].video_cache,self.BSs[1].video_cache,self.BSs[2].video_cache))
            # self.all_cache=self.video
            r=0
            for bs in self.BSs:
                reward=self.deal_each_bs(bs)
                r+=reward
            self.rewards.append(r)
        return sum(self.rewards)
    def get_avg_Qoe(self,):
        res=[]
        for i in range(para.test_times):
            self.rewards=[]
            r=self.get_QoE()
            res.append(r)
        print("Baseline2:",sum(res)/len(res))
        return sum(res)/len(res)
class Baseline3():   # 随机算法
    def __init__(self,BSs):
        self.rewards=[]
        self.BSs=BSs
        for b in self.BSs:
            b.video_cache=b.ini_state[0]
            b.all_video_hot=b.ini_state[1]
        pass
    def deal_each_bs(self, bs: envs.Basestation):
        prob = bs.all_video_hot
        # random_request = np.random.choice(np.arange(para.num_videos), size=para.request_times, p=prob)
        random_request =para.request(prob)
        rs = []
        for v in random_request:
            if v in bs.video_cache:
                r = 1
            elif v in self.all_cache:
                r = 1 - para.delay_n
            else:
                r = 1 - para.delay_Cloud
            rs.append(r)
        return sum(rs)
    def get_QoE(self,):
        for _ in range(para.Times):
            self.all_cache=np.concatenate((self.BSs[0].video_cache,self.BSs[1].video_cache,self.BSs[2].video_cache))
            self.all_cache=np.unique(self.all_cache)
            r=0
            for bs in self.BSs:
                reward=self.deal_each_bs(bs)
                r+=reward
            self.rewards.append(r)
            for i in range(para.N):
                action=np.random.randint(para.num_videos,size=3)
                self.BSs[i].get_next_obs(action[i])

        return sum(self.rewards)
    def get_avg_Qoe(self,):
        res=[]
        for i in range(para.test_times):
            self.rewards=[]
            r=self.get_QoE()
            res.append(r)
        print("Baseline3:",sum(res)/len(res))
        return sum(res)/len(res)
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1.5e3), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=para.Times, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name="simple_spread", number=1, seed=52)  #8978578
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    para.train=False
    pro=runner.test()

    BSs=runner.env.BSs

    b1=Baseline1()
    res1=b1.get_avg_Qoe()
    # print(res1)
    b2=Baseline2()
    res2=b2.get_avg_Qoe()

    b3=Baseline3()
    res3=b3.get_avg_Qoe()