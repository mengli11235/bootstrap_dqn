import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from run_bootstrap import rolling_average

def plots(xs, ys, xlabel, ylabel, title, legends, loc="lower right"):
    if not os.path.exists('figs'):
        os.makedirs('figs')
    for i,x in enumerate(xs):
        plt.plot(x,ys[i], linewidth=1.5,) #linestyle=(0, (i+3, 1, 2*i, 1)),)
    #plt.legend(loc=loc, ncol=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('figs', title + ".jpg"))
    plt.close()
    t = datetime.now().strftime("_%Y_%m_%d")

def plots_err(xs, ys, ystd, xlabel, ylabel, title, legends, loc="lower right", color=['b','y','g', 'r']):
    if not os.path.exists('figs'):
        os.makedirs('figs')
    for i,x in enumerate(xs):
        #plt.errorbar(x, ys[i], xerr=0.5, yerr=2*ystd[i], label=legends[i], color=color[i], linewidth=1.5,) #linestyle=(0, (i+3, 1, 2*i, 1)),)
        plt.plot(x,ys[i], color=color[i], linewidth=1.5,) #linestyle=(0, (i+3, 1, 2*i, 1)),)
        if True: #i==0:
            plt.fill_between(x, ys[i]-2*ystd[i], ys[i]+2*ystd[i], color=color[i], alpha=0.1)
    #plt.legend(loc=loc, ncol=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('figs', title + ".jpg"))
    plt.close()
    t = datetime.now().strftime("_%Y_%m_%d")

if __name__ == '__main__':
    game_name = 'qbert'
    path1 = 'model_savedir/' + game_name + '00/bestq.pkl'
    path2 = 'model_savedir/' + game_name + '01/bestq.pkl'

    model_dict1 = torch.load(path1).to('cpu')
    model_dict2 = torch.load(path2).to('cpu')

    info = model_dict1['info']
    perf1 = model_dict1['perf']
    perf2 = model_dict2['perf']

    steps1 = perf1['steps']
    steps 2 = perf2['steps']
    eval_steps1 = perf1['eval_steps']
    eval_steps2 = perf2['eval_steps']

    y1_mean_scores = perf1['eval_rewards']
    y1_std_scores = perf1['eval_stds']
    y1q = perf1['q_record']

    y2_mean_scores = perf2['eval_rewards']
    y2_std_scores = perf2['eval_stds']
    y2q = perf2['q_record']
    print(perf1['highest_eval_score'][-1], perf2['highest_eval_score'][-1])

    title = "Mean Evaluation Scores in "+ game_name
    legends = ['Boot-DQN*', 'Boot-DQN+PR']

    plots_err(
        [eval_steps1, eval_steps2],
        [y1_mean_scores, y2_mean_scores],
        [y1_std_scores, y2_std_scores],
        "Step",
        "Score",
        title,
        legends,
    )

    title = "Maximal Q-values in "+game_name
    plots(
        [steps1, steps2],
        [y1q, y2q],
        "Step",
        "Q value",
        title,
        legends,
        loc="upper left"
    )
