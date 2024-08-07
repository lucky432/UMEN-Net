
from email import header
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import scipy.io as io
import numpy as np

# 显示loss
def plot_curve(loss_list_G):
    """
    根据存放loss值的列表绘制曲线
    
    """
    fig, axes = plt.subplots(1,1, figsize=(5, 8))
    ax0 = axes.plot(range(len(loss_list_G)), loss_list_G, color = 'blue')
    axes.set_title('G_loss')

    # ax1 = axes[1].plot(range(len(loss_list_G)), loss_list_G, color = 'red') 
    # axes[1].set_title('G_loss')

    fig.savefig('/202222000580/mpwcnet++/results/reusenet_sim/loss_plot/160_ela_loss_vgg')

def read_loss(path_G):
    data_loss_G = np.load(path_G)
    return data_loss_G

# path_G = '/202222000580/mpwcnet++/results/reusenet_sim/RF_loss/300_total_loss.npy'
# data_loss_G=read_loss(path_G)
# # print(data_loss_G)
# plot_curve(data_loss_G)

