# time: 2024/1/26 21:20
# author: YanJP
import numpy as np
N=3
num_videos=15  # 一共有15份视频
cachelen=6  # 每个基站的视频存储长度
state_dim =2*num_videos    # 自己基站的视频缓存（设定所有基站共享视频，但有时延）、每个视频的请求热度（Zipf给出，用户按照这个概率请求视频）、
action_dim=num_videos   # action决策缓存视频，然后卸载最后一个（每个agent维护一个视频缓存列表，按请求热度排序）
Times=20

request_times=30
delay_n=0.3
delay_Cloud=0.8

def get_hot():
    h=np.random.uniform(0.1,0.9,size=num_videos)
    nor_hot=h/sum(h)
    return nor_hot

a=1.8
def get_hot_zipf():
    # 生成Zipf分布的概率
    # a =1.8  # Zipf分布的参数，可以调整   2.5
    zipf_probs = np.random.zipf(a, num_videos)
    # 归一化概率，使其总和为1
    zipf_probs_normalized = zipf_probs / np.sum(zipf_probs)
    # 打印生成的概率
    # print("生成的Zipf分布概率：", zipf_probs_normalized)
    return zipf_probs_normalized

def ini_video():
    videos_c=[]
    for i in range(3):
        video_cache = np.random.choice(np.arange(num_videos), size=cachelen, replace=False)
        videos_c.append(video_cache)
    all_video_hot = get_hot_zipf()

    return videos_c,all_video_hot

videos_c,all_video_hot=ini_video()

train=True
test_times=40


def request(prob):
    random_request = np.random.choice(np.arange(num_videos), size=request_times, p=prob)
    return random_request

if __name__ == '__main__':

    (get_hot_zipf())

