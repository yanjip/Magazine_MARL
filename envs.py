# time: 2024/1/26 21:16
# author: YanJP
import numpy as np
import para

class Basestation():
    def __init__(self,id):
        self.id=id
        self.video_cache=np.random.choice(np.arange(para.num_videos),size=para.cachelen,replace=False)
        # self.get_obs()
        self.all_video_hot=para.get_hot_zipf()
        self.video_hot=self.all_video_hot[self.video_cache]
    def sort(self,):
        # 根据概率值对视频ID进行排序   argsort()默认升序排序，所以热度越大的视频ID会排在后面
        sorted_indices = self.video_hot.argsort()
        # 使用索引获取排序后的视频ID
        self.video_cache = self.video_cache[sorted_indices]

    def update_hot(self,):
        # self.all_video_hot=para.get_hot_zipf()
        # self.video_hot=self.all_video_hot[self.video_cache]
        self.sort()
    def get_dummy(self,):
        zeros=np.zeros(para.num_videos)
        zeros[self.video_cache]=1
        self.dummy_video_cache=zeros
    def get_ini_obs(self,):
        self.update_hot()
        self.get_dummy()
        self.obs=np.concatenate((self.all_video_hot,self.dummy_video_cache),axis=0)
        return self.obs
    def get_next_obs(self,action):
        self.update_hot()  # 下一个时隙的热度
        if np.isin(action, self.video_cache):
            self.obs = np.concatenate((self.all_video_hot, self.dummy_video_cache), axis=0)
            # self.get_dummy()
            return self.obs  #下一个状态
        self.video_cache[0]=action
        self.sort()
        self.get_dummy()
        self.obs=np.concatenate((self.all_video_hot,self.dummy_video_cache),axis=0)
        return self.obs


class env_():
    def __init__(self):
        self.n=para.N
        self.action_space=para.action_dim
        self.observation_space=para.state_dim  #每个智能体的空间维度
        # self.UserAll=trans.generateU()
        self.reward=0
        self._max_episode_steps=para.Times

        self.BSs= [Basestation(0),Basestation(1),Basestation(2)]
        # self.h=para.h
        # self.min_simis=para.min_sims
        # self.salency=para.salency
        # self.request_tiles=np.random.randint(para.N_fov_low,para.N_fov)
        pass
    def get_all_obs(self,):
        ob0=self.BSs[0].get_ini_obs()
        ob1=self.BSs[1].get_ini_obs()
        ob2=self.BSs[2].get_ini_obs()
        return [ob0,ob1,ob2]

    def reset(self,):
        self.done=[0,0,0]
        self.index=0

        # self.Nc_left=para.N_c
        # self.Nc_left_norm = self.Nc_left/para.N_c
        # obs=np.concatenate((self.BSs[0],self.BSs[0],self.BSs[0],),axis=0)
        obs=self.get_all_obs()
        return obs
        # state：[time_step, carrier_left, tile_number]  加不加上tilenumber呢，这很有影响
    # def get_o(self,snr,simi_min):
    #     for oi in range(len(para.compress)):
    #         simi=fitting.fitting_func(oi,snr)
    #         if simi>=simi_min:
    #             return para.compress[oi]
    #     return 1.0
    def deal_each_bs(self,bs:Basestation):
        prob=bs.all_video_hot
        random_request=np.random.choice(np.arange(para.num_videos),size=para.request_times,p=prob)
        rs=[]
        for v in random_request:
            if v in bs.video_cache:
                r=0.01
            elif v in self.all_cache:
                r=0.01-para.delay_n
            else:
                r=0.01-para.delay_Cloud
            rs.append(r)
        return sum(rs)


    def step(self,action):
        #  一个大时隙里，模拟很多次用户请求 Reward设置为：成功1-时延；失败0
        # 模拟用户请求
        rewards=[]
        self.all_cache=np.concatenate((self.BSs[0].video_cache,self.BSs[1].video_cache,self.BSs[2].video_cache))
        self.all_cache=np.unique(self.all_cache)
        for bs in self.BSs:
            reward=self.deal_each_bs(bs)
            rewards.append(reward)

        self.index += 1
        next_obs=[]

        if self.index==para.Times:
            self.done=[1.0,1.,1.]
        for i in range(para.N):
            each_obs=self.BSs[i].get_next_obs(action[i])
            next_obs.append(each_obs)

        return next_obs, rewards, self.done, None



if __name__ == '__main__':
    b1=Basestation(1)
    print(b1.video_cache)
    print(b1.get_obs())
    print(b1.all_video_hot)
    print(b1.video_hot)
    print(b1.dummy_video_cache)
