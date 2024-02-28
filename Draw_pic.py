# time: 2023/12/23 21:36
# author: YanJP
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
# import torch
import pandas as pd
import para
import datetime
import pickle
from matplotlib.font_manager import FontProperties  # 导入字体模块

# 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
def chinese_font():
    try:
        font = FontProperties(
            # 系统字体路径
            fname='C:\\Windows\\Fonts\\方正粗黑宋简体.ttf', size=14)
    except:
        font = None
    return font
def process_res(res):
    proposed=res[:,0]
    b1=res[:,1]
    b2=res[:,2]
    b3=res[:,3]
    return proposed,b1,b2,b3
def plot_rewards(rewards,time,  path=None,):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title(f"{tag}ing curve on {cfg['device']} ")
    # plt.title("PPO Algorithm")
    plt.rc('font', size=15)
    plt.xlabel('Epsiodes', fontsize=17, fontweight='bold', labelpad=-1)
    plt.ylabel('Reward', fontsize=17, fontweight='bold', labelpad=-1)
    plt.grid(linestyle="--", color="gray", linewidth="0.5", axis="both")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    s_r1 = smooth(rewards)
    # s_r2=smooth(rewards[1])
    # s_r1=smooth(s_r1)
    # s_r2=smooth(s_r2)
    plt.plot(rewards,alpha=0.5,color='c')
    plt.plot(s_r1, linewidth='1.5', )
    # plt.plot(s_r2,linewidth='1.5', label='clipped probability ratio=0.5')
    # plt.ylim(0)

    # plt.legend()
    a = time
    plt.savefig('runs/pic/' + a + "_rewards.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.show()

def plot_BW(x,res):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    proposed, b1, b2, b3,b4 = process_res(res)
    plt.plot(x, proposed, marker='>', markersize=8, label='Proposed Scheme')  # 使用三角形节点
    plt.plot(x, b1, marker='o', markersize=8, label='Baseline 1',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b4, marker='*', markersize=8, label='Baseline 2',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b3, marker='d', markersize=8, label='Baseline 3',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b2, marker='s', markersize=8, label='Baseline 4',markerfacecolor='none')  # 使用三角形节点
    plt.rc('font', size=17)
    plt.legend(loc='lower right', ncol=1)
    # plt.ylim(2)
    plt.grid(linestyle="--",color="gray",linewidth="0.5",axis="both")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel('Bandwidth (MHz)', fontsize=17,fontweight='bold',labelpad=-2)
    plt.ylabel('QoE', fontsize=17,fontweight='bold',labelpad=-10)
    # plt.title('Total QoE at Time Slot')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/new_baseline/Bandwidth" + a, dpi=600, bbox_inches='tight', pad_inches=0.01)
    # 显示图形
    plt.show()

def plot_power(x,res):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    proposed, b1, b2, b3 = process_res(res)
    plt.plot(x, proposed, marker='>', markersize=8, label='Proposed Scheme')  # 使用三角形节点
    plt.plot(x, b1, marker='o', markersize=8, label='Baseline 1',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b2, marker='d', markersize=8, label='Baseline 2',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b3, marker='s', markersize=8, label='Baseline 3',markerfacecolor='none')  # 使用三角形节点
    plt.rc('font', size=13)
    # plt.legend(loc='lower right', ncol=1)
    plt.legend(ncol=2)
    plt.ylim(0.01,0.20)
    plt.grid(linestyle="--",color="gray",linewidth="0.5",axis="both")
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel(r'Maximum Power (W)', fontsize=17,fontweight='bold',labelpad=0)
    plt.ylabel('Average Time Delay (s)', fontsize=17,fontweight='bold',labelpad=0)
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/Power" + a, dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.show()


# 用于平滑曲线，类似于Tensorboard中的smooth
def smooth(data, weight=0.9):
    '''
    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_rewards_from_file(rewards,times, path=None,):
    # sns.set()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title(f"{tag}ing curve on {cfg['device']} ")
    # plt.title("PPO Algorithm")
    plt.rc('font', size=15)
    plt.xlabel('Epsiodes', fontsize=17,fontweight='bold',labelpad=-1)
    plt.ylabel('Reward', fontsize=17,fontweight='bold',labelpad=-1)
    plt.grid(linestyle="--",color="gray",linewidth="0.5",axis="both")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    s_r1=smooth(rewards)
    # s_r2=smooth(rewards[1])
    # s_r1=smooth(s_r1)
    # s_r2=smooth(s_r2)
    # plt.plot(rewards,alpha=0.5,color='g')
    plt.plot(s_r1, linewidth='1.5', )
    # plt.plot(s_r2,linewidth='1.5', label='clipped probability ratio=0.5')
    # plt.ylim(0)

    # plt.legend()
    a = times
    plt.savefig('runs/simulation_res/'+a+"_rewards.png",dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.show()
def pic_reward():
    # r1=np.load('runs/rewards/2023_12_25-15_27_51_reward.npy')
    # r2=np.load('runs/rewards/2023_12_24-21_16_04_reward.npy')

    r1=np.load("runs/reward/2024_01_27-20_44_29_reward.npy")
    # r2=np.load('runs/rewards/2023_12_16-15_08_24_reward.npy')

    plot_rewards_from_file(r1,times='1_27')



def plot_Qoe_bar(x,res):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rc('font', size=13)
    plt.grid(linestyle="--", color="gray", linewidth="0.5", axis="both")
    proposed, b1, b2, b3 = process_res(res)
    categories =[1.8,2.0,2.2,2.4,2.6]
    x = np.arange(len(categories))
    # 使用plt.bar()替代plt.plot()
    width = 0.2  # 设置柱状图的宽度

    plt.bar(x, proposed, width=width, label='Proposed Scheme')
    plt.bar([i + width for i in x], b1, width=width, label='Baseline 1', alpha=0.7)
    plt.bar([i + 2* width for i in x], b3, width=width, label='Baseline 3', alpha=0.7)
    plt.bar([i + 3 * width for i in x], b2, width=width, label='Baseline 2', alpha=0.7)
    plt.ylim(1000,1700)


    plt.xticks([i + 1.5 * width for i in x], categories)  # 调整x轴刻度位置
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel(r'$\gamma$', fontsize=17, labelpad=0)
    plt.ylabel('QoE', fontsize=17, labelpad=1)
    plt.legend(ncol=2)

    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/Qoe" + a, dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.show()

def rainbow(data1,data2):

    sr1=smooth(data1)
    sr2=smooth(data2)
    data=[sr1,sr2]

    y_mean = np.mean((np.array(data)), axis=0)
    y_std = np.std((np.array(data)), axis=0)
    y_max = y_mean + y_std * 0.99
    y_min = y_mean - y_std * 0.99

    x = np.arange(0, len(data1), 1)

    fig = plt.figure(1)
    plt.plot(x, y_mean, label='method1', color='#e75840')
    plt.fill_between(x, y_max, y_min, alpha=0.5, facecolor='#e75840')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == '__main__':
    data1 = np.load("runs/reward/2024_01_27-20_44_29_reward.npy")
    data2 = np.load("runs/reward/2024_01_28-13_31_52_reward.npy")
    rainbow(data1,data2)

    # pic_reward()


    # zs=[1.8,2.0,2.2,2.4,2.6]
    # res = np.load('runs/simulation_res/Qoe.npy')
    # plot_Qoe_bar(zs,res)

    pass