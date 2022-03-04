import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from run_bootstrap import rolling_average
import string

def plots(xs, ys, xlabel, ylabel, title, legends, loc="lower right", color=['b','y','g', 'r']):
    if not os.path.exists('figs'):
        os.makedirs('figs')
    for i,x in enumerate(xs):
        plt.plot(x,ys[i], linewidth=1.5,color=color[i],) #linestyle=(0, (i+3, 1, 2*i, 1)),)
    #plt.legend(loc=loc, ncol=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('figs', title + ".jpg"))
    plt.close()

def plots_err(xs, ys, ystd, xlabel, ylabel, title, legends, loc="lower right", color=['b','y','g', 'r']):

    if not os.path.exists('figs'):
        os.makedirs('figs')
    for i,x in enumerate(xs):
        #plt.errorbar(x, ys[i], xerr=0.5, yerr=2*ystd[i], label=legends[i], color=color[i], linewidth=1.5,) #linestyle=(0, (i+3, 1, 2*i, 1)),)
        plt.plot(x,ys[i], color=color[i], linewidth=1.5,) #linestyle=(0, (i+3, 1, 2*i, 1)),)
        if True: #i==0:
            plt.fill_between(x, np.array(ys[i])-2*np.array(ystd[i]), np.array(ys[i])+2*np.array(ystd[i]), color=color[i], alpha=0.1)
    #plt.legend(loc=loc, ncol=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('figs', title + ".jpg"))
    plt.close()

if __name__ == '__main__':
    game_name = 'krull'
    titile_name = string.capwords(game_name.replace("_", " "))
    path1 = '../bootstrap_results/model_savedir/' + game_name + '00/'+game_name+'_bestq.pkl'
    path2 = '../bootstrap_results/model_savedir/' + game_name + '01/'+game_name+'_bestq.pkl'

    model_dict1 = torch.load(path1, map_location=torch.device('cpu'))
    model_dict2 = torch.load(path2, map_location=torch.device('cpu'))

    info = model_dict1['info']
    perf1 = model_dict1['perf']
    perf2 = model_dict2['perf']

    steps1 = perf1['steps']
    steps2 = perf2['steps']
    eval_steps1 = perf1['eval_steps']
    eval_steps2 = perf2['eval_steps']

    y1_mean_scores = perf1['eval_rewards']
    y1_std_scores = perf1['eval_stds']
    y1q = perf1['q_record']

    y2_mean_scores = perf2['eval_rewards']
    y2_std_scores = perf2['eval_stds']
    y2q = perf2['q_record']
    print(perf1['highest_eval_score'][-1], perf2['highest_eval_score'][-1])

    title = "Mean Evaluation Scores in "+ titile_name
    legends = ['Boot-DQN*', 'Boot-DQN+PR']

    plots_err(
        [eval_steps1, eval_steps2],
        [y1_mean_scores, y2_mean_scores],
        [y1_std_scores, y2_std_scores],
        "Steps",
        "Scores",
        title,
        legends,
    )

    # plots(
    #     [eval_steps1, eval_steps2],
    #     [y1_mean_scores, y2_mean_scores],
    #     "Steps",
    #     "Scores",
    #     title,
    #     legends,
    #     loc="upper left"
    # )

    title = "Maximal Q-values in "+ titile_name
    plots(
        [steps1, steps2],
        [y1q, y2q],
        "Steps",
        "Q values",
        title,
        legends,
        loc="upper left"
    )
