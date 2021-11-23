from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
from IPython import embed
from collections import Counter
import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
from dqn_model import EnsembleNet, NetWithPrior
from dqn_utils import seed_everything, write_info_file, generate_gif, save_checkpoint
from env import Environment
from replay import ReplayMemory
import config
from diayn_policies import *

def rolling_average(a, n=5) :
    if n == 0:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_dict_losses(plot_dict, name='loss_example.png', rolling_length=4, plot_title=''):
    f,ax=plt.subplots(1,1,figsize=(6,6))
    for n in plot_dict.keys():
        print('plotting', n)
        ax.plot(rolling_average(plot_dict[n]['index']), rolling_average(plot_dict[n]['val']), lw=1)
        ax.scatter(rolling_average(plot_dict[n]['index']), rolling_average(plot_dict[n]['val']), label=n, s=3)
    ax.legend()
    if plot_title != '':
        plt.title(plot_title)
    plt.savefig(name)
    plt.close()

def matplotlib_plot_all(p):
    epoch_num = len(p['steps'])
    epochs = np.arange(epoch_num)
    steps = p['steps']
    plot_dict_losses({'episode steps':{'index':epochs,'val':p['episode_step']}}, name=os.path.join(model_base_filedir, 'episode_step.png'), rolling_length=0)
    plot_dict_losses({'episode steps':{'index':epochs,'val':p['episode_relative_times']}}, name=os.path.join(model_base_filedir, 'episode_relative_times.png'), rolling_length=10)
    plot_dict_losses({'episode head':{'index':epochs, 'val':p['episode_head']}}, name=os.path.join(model_base_filedir, 'episode_head.png'), rolling_length=0)

    episode_loss_mask = np.isfinite(p['episode_loss'])
    plot_dict_losses({'steps loss':{'index':np.array(steps)[episode_loss_mask], 'val':np.array(p['episode_loss'])[episode_loss_mask]}}, name=os.path.join(model_base_filedir, 'steps_loss.png'))

    plot_dict_losses({'steps eps':{'index':steps, 'val':p['eps_list']}}, name=os.path.join(model_base_filedir, 'steps_mean_eps.png'), rolling_length=0)
    plot_dict_losses({'steps reward':{'index':steps,'val':p['episode_reward']}},  name=os.path.join(model_base_filedir, 'steps_reward.png'), rolling_length=0)
    plot_dict_losses({'episode reward':{'index':epochs, 'val':p['episode_reward']}}, name=os.path.join(model_base_filedir, 'episode_reward.png'), rolling_length=0)
    plot_dict_losses({'episode times':{'index':epochs,'val':p['episode_times']}}, name=os.path.join(model_base_filedir, 'episode_times.png'), rolling_length=5)
    plot_dict_losses({'steps avg reward':{'index':steps,'val':p['avg_rewards']}}, name=os.path.join(model_base_filedir, 'steps_avg_reward.png'), rolling_length=0)

    eval_steps_mask = np.isfinite(p['eval_steps'])
    eval_rewards_mask = np.isfinite(p['eval_rewards'])

    plot_dict_losses({'eval rewards':{'index':np.array(p['eval_steps'])[eval_steps_mask], 'val':np.array(p['eval_rewards'])[eval_rewards_mask]}}, name=os.path.join(model_base_filedir, 'eval_rewards_steps.png'), rolling_length=0)

    #plot_dict_losses({'eval states':{'index':np.array(p['eval_steps'])[eval_steps_mask], 'val':np.array(p['eval_num_states'])[eval_rewards_mask]}}, name=os.path.join(model_base_filedir, 'eval_num_states_steps.png'), rolling_length=0)

def handle_checkpoint(last_save, cnt):
    if (cnt-last_save) >= info['CHECKPOINT_EVERY_STEPS']:
        st = time.time()
        print("beginning checkpoint", st)
        last_save = cnt
        state = {'info':info,
                 'optimizer':opt.state_dict(),
                 'cnt':cnt,
                 'policy_net_state_dict':policy_net.state_dict(),
                 'target_net_state_dict':target_net.state_dict(),
                 'perf':perf,
                }
        filename = os.path.abspath(model_base_filepath + "_%010dq.pkl"%cnt)
        if torch.cuda.is_available():
            filename = os.path.abspath("/scratch/users/limeng/buffer" + "_%010dq.pkl"%cnt)
        save_checkpoint(state, filename)
        # npz will be added
        buff_filename = os.path.abspath(model_base_filepath + "_%010dq_train_buffer"%cnt)
        if torch.cuda.is_available():
                    buff_filename = os.path.abspath("/scratch/users/limeng/buffer" + "_%010dq_train_buffer"%cnt)
        #replay_memory.save_buffer(buff_filename)
        print("finished checkpoint", time.time()-st)
        return last_save
    else: return last_save


class ActionGetter:
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""
    """This class is from fg91's dqn. TODO put my function back in"""
    def __init__(self, n_actions, eps_initial=1, eps_final=0.1, eps_final_frame=0.01,
                 eps_evaluation=0.0, eps_annealing_frames=100000,
                 replay_memory_start_size=50000, max_steps=25000000, random_seed=122):
        """
        Args:
            n_actions: Integer, number of possible actions
            eps_initial: Float, Exploration probability for the first
                replay_memory_start_size frames
            eps_final: Float, Exploration probability after
                replay_memory_start_size + eps_annealing_frames frames
            eps_final_frame: Float, Exploration probability after max_frames frames
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_frames: Int, Number of frames over which the
                exploration probabilty is annealed from eps_initial to eps_final
            replay_memory_start_size: Integer, Number of frames during
                which the agent only explores
            max_frames: Integer, Total number of frames shown to the agent
        """
        self.n_actions = n_actions
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.replay_memory_start_size = replay_memory_start_size
        self.max_steps = max_steps
        self.random_state = np.random.RandomState(random_seed)

        # Slopes and intercepts for exploration decrease
        if self.eps_annealing_frames > 0:
            self.slope = -(self.eps_initial - self.eps_final)/self.eps_annealing_frames
            self.intercept = self.eps_initial - self.slope*self.replay_memory_start_size
            self.slope_2 = -(self.eps_final - self.eps_final_frame)/(self.max_steps - self.eps_annealing_frames - self.replay_memory_start_size)
            self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_steps

    def pt_get_action(self, step_number, state, active_head=None, evaluation=False):
        """
        Args:
            step_number: int number of the current step
            state: A (4, 84, 84) sequence of frames of an atari game in grayscale
            active_head: number of head to use, if None, will run all heads and vote
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
            An integer between 0 and n_actions
        """
        #state = torch.Tensor(state.astype(np.float)/info['NORM_BY'])[None,:].to(info['DEVICE'])
        # if 'discriminator' in info['IMPROVEMENT']:
        #     logits = discriminator(state, 0)
        #     #logits = torch.softmax(discriminator(states, 0), dim=-1)
        #     action_head = torch.argmax(logits, dim=-1).item()

        if evaluation:
            eps = self.eps_evaluation
        elif step_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif self.eps_annealing_frames > 0:
            # TODO check this
            if step_number >= self.replay_memory_start_size and step_number < self.replay_memory_start_size + self.eps_annealing_frames:
                eps = self.slope*step_number + self.intercept
            elif step_number >= self.replay_memory_start_size + self.eps_annealing_frames:
                eps = self.slope_2*step_number + self.intercept_2
        else:
            eps = 0
        if self.random_state.rand() < eps:
            return eps, self.random_state.randint(0, self.n_actions)
        else:
            state = torch.Tensor(state.astype(np.float)/info['NORM_BY'])[None,:].to(info['DEVICE'])
            if not evaluation and ('SURGE' in info['IMPROVEMENT'] or 'SURGE_OUT' in info['IMPROVEMENT']):
                vals = policy_net(state, None)
                acts = [torch.argmax(vals[h],dim=1).item() for h in range(active_head+1)]
                if len(acts) == 1:
                    acts = acts[0]
                return eps, acts
            else:
                vals = policy_net(state, active_head)
                if active_head is not None:
                    action = torch.argmax(vals, dim=1).item()
                    return eps, action
                else:
                    # vote
                    acts = [torch.argmax(vals[h],dim=1).item() for h in range(info['N_ENSEMBLE'])]
                    data = Counter(acts)
                    action = data.most_common(1)[0][0]
                    heads_chosen = [0]*info['N_ENSEMBLE']
                    for i,head in enumerate(acts):
                        if action == head:
                            heads_chosen[i] += 1

                        return heads_chosen, action

def ptlearn(states, actions, rewards, next_states, terminal_flags, active_heads, masks):
    states = torch.Tensor(states.astype(np.float)/info['NORM_BY']).to(info['DEVICE'])
    next_states = torch.Tensor(next_states.astype(np.float)/info['NORM_BY']).to(info['DEVICE'])
    rewards = torch.Tensor(rewards).to(info['DEVICE'])
    actions = torch.LongTensor(actions).to(info['DEVICE'])
    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(info['DEVICE'])
    active_heads = torch.LongTensor(active_heads).to(info['DEVICE'])
    masks = torch.FloatTensor(masks.astype(np.int)).to(info['DEVICE'])
    # min history to learn is 200,000 frames in dqn - 50000 steps
    losses = [0.0 for _ in range(info['N_ENSEMBLE'])]

    opt.zero_grad()
    q_policy_vals = policy_net(states, None)
    next_q_target_vals = target_net(next_states, None)
    next_q_policy_vals = policy_net(next_states, None)

    cnt_losses = []
    if 'DISCRIMINATOR' in info['IMPROVEMENT']:
        opt_discriminator.zero_grad()
        logits = torch.softmax(discriminator(states, 0), dim=-1)
        masks = 1-logits.detach()
        #print(active_heads, logits)
        # next_logits = torch.softmax(discriminator(next_states, 0), dim=-1)

        # prior_pi = (1-logits.detach()).transpose(0,1)
        # prior_next_pi = (1-next_logits.detach()).transpose(0,1)
        discriminator_loss = ce_loss(logits, active_heads)


    if 'PRETRAIN' in info['IMPROVEMENT']:
        if 'PRIOR' in info['IMPROVEMENT']:
            prior_pi = prior_net(states, None)
            prior_next_pi = prior_net(next_states, None)
        else:
            prior_pi = prior_net.forward(states, return_all_heads=True)
            prior_next_pi  = prior_net.forward(next_states, return_all_heads=True)
        #print(prior_next_pi[0], next_q_target_vals[0])
#         q_policy_vals += info['PRIOR_SCALE'] * prior_pi
#         next_q_target_vals += info['PRIOR_SCALE'] * prior_next_pi

    if 'entropy' in info['IMPROVEMENT']:
        prior_q_policy_vals = policy_net.return_prior(states, None)
        prior_next_q_target_vals = target_net.return_prior(next_states, None)
        prior_next_q_policy_vals = policy_net.return_prior(next_states, None)
    for k in range(info['N_ENSEMBLE']):
        #TODO finish masking
        total_used = torch.sum(masks[:,k])
        if total_used > 0.0:
            next_k = k
            if ('SURGE' in info['IMPROVEMENT'] or 'SURGE_OUT' in info['IMPROVEMENT']) and k > 0:
                next_k = k-1
            next_q_vals = next_q_target_vals[next_k].data
            if info['DOUBLE_DQN']:
                next_actions = next_q_policy_vals[next_k].data.max(1, True)[1]
                next_qs = next_q_vals.gather(1, next_actions).squeeze(1)
            else:
                next_qs = next_q_vals.max(1)[0] # max returns a pair

            preds = q_policy_vals[k].gather(1, actions[:,None]).squeeze(1) 
            # if k==0:
            #     print(q_policy_vals[k])

            if 'PRETRAIN' in info['IMPROVEMENT']:
                if 'PRETRAIN' in info['IMPROVEMENT']:
                    prior_preds = prior_pi[k].gather(1, actions[:,None]).squeeze(1)
                    next_prior_preds = prior_next_pi[k].gather(1, next_actions).squeeze(1)
                else:
                    prior_preds = prior_pi[k]
                    next_prior_preds = prior_next_pi[k]
                preds += info['PRIOR_SCALE'] * prior_preds
                if not info['DOUBLE_DQN']:
                    next_actions = torch.argmax(next_q_vals, dim=1)
                next_qs += info['PRIOR_SCALE'] * next_prior_preds

            targets = info['GAMMA'] * next_qs * (1-terminal_flags)
            if not ('SURGE' in info['IMPROVEMENT'] or 'SURGE_OUT' in info['IMPROVEMENT']) or k == 0:
                targets += rewards
            l1loss = F.smooth_l1_loss(preds, targets, reduction='mean')
            # if 'soft' in info['IMPROVEMENT']:
            #     # soft update
            #     #soft_prior_loss = 4 * torch.log(torch.sum(torch.exp(prior_q_policy_vals[k]/4), dim=-1))
            #     soft_prior_loss =torch.sum(torch.exp(prior_q_policy_vals[k]/4), dim=-1))

            if 'entropy' in info['IMPROVEMENT']:
                # # loss of H(a|s,z)
#                 logits = torch.softmax(prior_q_policy_vals[k], dim=-1) #batch*a
#                 logits = torch.sum(logits*torch.log(logits), dim=-1) #batch
#                 l1loss += 0.001*logits.mean() #1
                #preds = 4 * torch.log(torch.sum(torch.exp(prior_q_policy_vals[k]/4), dim=-1))
                preds = 1*prior_q_policy_vals[k].gather(1, actions[:,None]).squeeze(1)


                #next_qs = 4 * torch.log(torch.sum(torch.exp(prior_next_q_target_vals[k].data/4), dim=-1))
                prior_next_actions = prior_next_q_policy_vals[k].data.max(1, True)[1]

                next_qs = prior_next_q_target_vals[k].data.gather(1, prior_next_actions).squeeze(1)

                #targets = -discriminator_loss.detach() + info['GAMMA'] * next_qs * (1-terminal_flags)
                targets = logits[:,k].detach() + info['GAMMA'] * next_qs * (1-terminal_flags)

                l1loss += F.smooth_l1_loss(preds, targets)
                #l1loss += kl_loss(torch.softmax(prior_q_policy_vals[k], dim=-1),torch.softmax(prior_next_q_target_vals[k].data, dim=-1))-discriminator_loss.detach()

            full_loss = masks[:,k]*l1loss #batch*1
            loss = torch.sum(full_loss/total_used)
            cnt_losses.append(loss)
            losses[k] = loss.cpu().detach().item()

    loss = sum(cnt_losses)/info['N_ENSEMBLE']
    loss.backward()
    for param in policy_net.core_net.parameters():
        if param.grad is not None:
            # divide grads in core
            param.grad.data *=1.0/float(info['N_ENSEMBLE'])
    nn.utils.clip_grad_norm_(policy_net.parameters(), info['CLIP_GRAD'])

    opt.step()
    if 'DISCRIMINATOR' in info['IMPROVEMENT']:
        discriminator_loss.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), info['CLIP_GRAD'])
        opt_discriminator.step()
    return np.mean(losses)#+discriminator_loss.detach().item()

def train(step_number, last_save):
    """Contains the training and evaluation loops"""
    epoch_num = len(perf['steps'])
    highest_eval_score = -np.inf
    waves = 0
    epoch_frame_episode_last = 0

    while step_number < info['MAX_STEPS']:
        ########################
        ####### Training #######
        ########################
        epoch_frame = 0
        while epoch_frame < info['EVAL_FREQUENCY']:
            terminal = False
            life_lost = True
            state = env.reset()
            start_steps = step_number
            st = time.time()
            episode_reward_sum = 0
            epoch_frame_episode = 0
            if 'DISCRIMINATOR' in info['IMPROVEMENT'] and step_number > info['MIN_HISTORY_TO_LEARN']:
                logits = discriminator(torch.Tensor(state.astype(np.float)/info['NORM_BY'])[None,:].to(info['DEVICE']), 0).detach()
                active_head = torch.argmin(logits, dim=-1).item()
            # elif 'SURGE' in info['IMPROVEMENT']:
            #     active_head = waves
            else:
                random_state.shuffle(heads)
                active_head = heads[0]
            epoch_num += 1
            ep_eps_list = []
            ptloss_list = []
            action_list = []
            while not terminal:
                if life_lost:
                    action = 1
                    eps = 0
                else:
                    if 'SURGE' in info['IMPROVEMENT']:
                        active_head = 0
                        if waves > 0 and step_number > info['MIN_HISTORY_TO_LEARN']:
                            active_head = waves - int(epoch_frame_episode/(epoch_frame_episode_last/(waves+1)))
                            if active_head < 0:
                                active_head = 0
                    elif 'SURGE_OUT' in info['IMPROVEMENT']:
                        active_head = 0
                        if step_number > info['MIN_HISTORY_TO_LEARN']:
                            active_head = waves

                    # if 'DISCRIMINATOR' in info['IMPROVEMENT'] and step_number > info['MIN_HISTORY_TO_LEARN']:
                    #     logits = discriminator(torch.Tensor(state.astype(np.float)/info['NORM_BY'])[None,:].to(info['DEVICE']), 0).detach()
                    #     active_head = torch.argmin(logits, dim=-1).item()
                    if ('SURGE' not in info['IMPROVEMENT'] and 'SURGE_OUT' not in info['IMPROVEMENT']) or len(action_list) == 0:
                        eps,action = action_getter.pt_get_action(step_number, state=state, active_head=active_head)
                        if not np.isscalar(action):
                            action_list = action
                            action = action_list.pop(0)

                    else:
                        if len(ep_eps_list):
                            eps = ep_eps_list[-1]
                        else:
                            eps = info['EPS_INITIAL']

                ep_eps_list.append(eps)
                next_state, reward, life_lost, terminal = env.step(action)
                # Store transition in the replay memory
                replay_memory.add_experience(action=action,
                                                frame=next_state[-1],
                                                reward=np.sign(reward), # TODO -maybe there should be +1 here
                                                terminal=life_lost,
                                                active_head= active_head)

                step_number += 1
                epoch_frame += 1
                epoch_frame_episode += 1
                episode_reward_sum += reward
                state = next_state

                if step_number % info['LEARN_EVERY_STEPS'] == 0 and step_number > info['MIN_HISTORY_TO_LEARN']:
                    _states, _actions, _rewards, _next_states, _terminal_flags, _active_heads, _masks = replay_memory.get_minibatch(info['BATCH_SIZE'])
                    ptloss = ptlearn(_states, _actions, _rewards, _next_states, _terminal_flags, _active_heads, _masks)
                    ptloss_list.append(ptloss)


                if 'SURGE' in info['IMPROVEMENT'] and step_number%info['SURGE_INTERVAL'] == 0 and step_number > info['MIN_HISTORY_TO_LEARN'] and waves < info['N_ENSEMBLE']-1:
                    waves += 1
                elif 'SURGE_OUT' in info['IMPROVEMENT'] and step_number%info['SURGE_INTERVAL'] == 0 and step_number > info['MIN_HISTORY_TO_LEARN']:
                    waves = (waves+1)%info['N_ENSEMBLE']

                if step_number % info['TARGET_UPDATE'] == 0 and step_number >  info['MIN_HISTORY_TO_LEARN']:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++")
                    print('updating target network at %s'%step_number)
                    target_net.load_state_dict(policy_net.state_dict())
                    if 'NORMAL_PRIOR' in info['IMPROVEMENT']:
                        prior_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                            n_actions=env.num_actions,
                            network_output_size=info['NETWORK_INPUT_SIZE'][0],
                            num_channels=info['HISTORY_SIZE'], dueling=False).to(info['DEVICE'])
                    #prior_target_net.load_state_dict(prior_net.state_dict())

            et = time.time()
            ep_time = et-st
            epoch_frame_episode_last = epoch_frame_episode

            perf['steps'].append(step_number)
            perf['episode_step'].append(step_number-start_steps)
            perf['episode_head'].append(active_head)
            perf['eps_list'].append(np.mean(ep_eps_list))
            perf['episode_loss'].append(np.mean(ptloss_list))
            perf['episode_reward'].append(episode_reward_sum)
            perf['episode_times'].append(ep_time)
            perf['episode_relative_times'].append(time.time()-info['START_TIME'])
            perf['avg_rewards'].append(np.mean(perf['episode_reward'][-100:]))
            last_save = handle_checkpoint(last_save, step_number)

            if not epoch_num%info['PLOT_EVERY_EPISODES'] and step_number > info['MIN_HISTORY_TO_LEARN']:
                # TODO plot title
                print('avg reward', perf['avg_rewards'][-1])
                print('last rewards', perf['episode_reward'][-info['PLOT_EVERY_EPISODES']:])

                matplotlib_plot_all(perf)
#                 with open('rewards.txt', 'a') as reward_file:
#                     print(len(perf['episode_reward']), step_number, perf['avg_rewards'][-1], file=reward_file)
        avg_eval_reward, highest_eval_score = evaluate(step_number, highest_eval_score)
        perf['eval_rewards'].append(avg_eval_reward)
        #perf['eval_num_states'].append(len(eval_states))
        perf['eval_steps'].append(step_number)
        matplotlib_plot_all(perf)

def evaluate(step_number, highest_eval_score):
    print("""
         #########################
         ####### Evaluation ######
         #########################
         """)
    eval_rewards = []
    evaluate_step_number = 0
    frames_for_gif = []
    heads_chosen = [0]*info['N_ENSEMBLE']

    # only run one
    for i in range(info['NUM_EVAL_EPISODES']):
        state = env.reset()
        episode_reward_sum = 0
        terminal = False
        life_lost = True
        episode_steps = 0
        while not terminal:
            if life_lost:
                action = 1
            else:
                active_head=None
                # if 'DISCRIMINATOR' in info['IMPROVEMENT']:
                #     logits = discriminator(torch.Tensor(state.astype(np.float)/info['NORM_BY'])[None,:].to(info['DEVICE']), 0).detach()
                #     action_head = torch.argmax(logits, dim=-1).item()
                if 'SURGE' in info['IMPROVEMENT'] or 'SURGE_OUT' in info['IMPROVEMENT']:
                    active_head = 0
                eps,action = action_getter.pt_get_action(step_number, state, active_head=active_head, evaluation=True)
                if 'SURGE' not in info['IMPROVEMENT'] and 'SURGE_OUT' not in info['IMPROVEMENT']:
                    heads_chosen = [x+y for x,y in zip(heads_chosen, eps)]
            next_state, reward, life_lost, terminal = env.step(action)
            # if next_state[-1].tobytes() not in eval_states:
            #     eval_states.append(next_state[-1].tobytes())
            evaluate_step_number += 1
            episode_steps +=1
            episode_reward_sum += reward
            # only save first episode
            frames_for_gif.append(env.ale.getScreenRGB())
            if not episode_steps%100:
                print('eval', episode_steps, episode_reward_sum)
            state = next_state
        eval_rewards.append(episode_reward_sum)
        if episode_reward_sum > highest_eval_score:
            highest_eval_score = episode_reward_sum
            generate_gif(model_base_filedir, 0, frames_for_gif, 0, name='test')
        frames_for_gif = []


    print("Evaluation score:\n", np.mean(eval_rewards))

    # Show the evaluation score in tensorboard
    efile = os.path.join(model_base_filedir, 'eval_rewards.txt')
    with open(efile, 'a') as eval_reward_file:
        print(step_number, np.mean(eval_rewards), heads_chosen, file=eval_reward_file)
    return np.mean(eval_rewards), highest_eval_score

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-l', '--model_loadpath', default='', help='.pkl model file full path')
    parser.add_argument('-b', '--buffer_loadpath', default='', help='.npz replay buffer file full path')
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("running on %s"%device)

    info = {
        #"GAME":'roms/breakout.bin', # gym prefix
        "GAME":'roms/breakout.bin', # gym prefix
        "DEVICE":device, #cpu vs gpu set by argument
        "NAME":'FRANKbootstrap_fasteranneal_pong', # start files with name
        "PRETRAIN_MODEL_PATH":'diayn_net_breakout', # start files with name
        "DUELING":True, # use dueling dqn
        "DOUBLE_DQN":True, # use double dqn
        "PRIOR":True, # turn on to use randomized prior
        "PRIOR_SCALE":0.1, # what to scale prior by
        "N_ENSEMBLE":9, # number of bootstrap heads to use. when 1, this is a normal dqn
        "LEARN_EVERY_STEPS":4, # updates every 4 steps in osband
        "BERNOULLI_PROBABILITY": 0.9, # Probability of experience to go to each head - if 1, every experience goes to every head
        "TARGET_UPDATE":10000, # how often to update target network
        "MIN_HISTORY_TO_LEARN":50000, # in environment frames
        "NORM_BY":255.,  # divide the float(of uint) by this number to normalize - max val of data is 255
        "EPS_INITIAL":1.0, # should be 1
        "EPS_FINAL":0.01, # 0.01 in osband
        "EPS_EVAL":0.0, # 0 in osband, .05 in others....
        "EPS_ANNEALING_FRAMES":int(1e6), # this may have been 1e6 in osband
        #"EPS_ANNEALING_FRAMES":0, # if it annealing is zero, then it will only use the bootstrap after the first MIN_EXAMPLES_TO_LEARN steps which are random
        "EPS_FINAL_FRAME":0.01,
        "NUM_EVAL_EPISODES":1, # num examples to average in eval
        "BUFFER_SIZE":int(1e6), # Buffer size for experience replay
        "CHECKPOINT_EVERY_STEPS":5000000, # how often to write pkl of model and npz of data buffer
        "EVAL_FREQUENCY":250000, # how often to run evaluation episodes
        "ADAM_LEARNING_RATE":6.25e-5,
        "RMS_LEARNING_RATE": 0.00025, # according to paper = 0.00025
        "RMS_DECAY":0.95,
        "RMS_MOMENTUM":0.0,
        "RMS_EPSILON":0.00001,
        "RMS_CENTERED":True,
        "HISTORY_SIZE":4, # how many past frames to use for state input
        "N_EPOCHS":90000,  # Number of episodes to run
        "BATCH_SIZE":32, # Batch size to use for learning
        "GAMMA":.99, # Gamma weight in Q update
        "PLOT_EVERY_EPISODES": 50,
        "CLIP_GRAD":5, # Gradient clipping setting
        "SEED":101,
        "RANDOM_HEAD":-1, # just used in plotting as demarcation
        "NETWORK_INPUT_SIZE":(84,84),
        "START_TIME":time.time(),
        "MAX_STEPS":int(50e6), # 50e6 steps is 200e6 frames
        "MAX_EPISODE_STEPS":27000, # Orig dqn give 18k steps, Rainbow seems to give 27k steps
        "FRAME_SKIP":4, # deterministic frame skips to match deepmind
        "MAX_NO_OP_FRAMES":30, # random number of noops applied to beginning of each episode
        "DEAD_AS_END":True, # do you send finished=true to agent while training when it loses a life
        "SURGE_INTERVAL":2e5,
        "IMPROVEMENT": ['NORMAL_PRIOR'],
    }

    info['FAKE_ACTS'] = [info['RANDOM_HEAD'] for x in range(info['N_ENSEMBLE'])]
    info['args'] = args
    info['load_time'] = datetime.date.today().ctime()
    info['NORM_BY'] = float(info['NORM_BY'])

    # create environment
    env = Environment(rom_file=info['GAME'], frame_skip=info['FRAME_SKIP'],
                      num_frames=info['HISTORY_SIZE'], no_op_start=info['MAX_NO_OP_FRAMES'], rand_seed=info['SEED'],
                      dead_as_end=info['DEAD_AS_END'], max_episode_steps=info['MAX_EPISODE_STEPS'])

    # create replay buffer
    replay_memory = ReplayMemory(size=info['BUFFER_SIZE'],
                                 frame_height=info['NETWORK_INPUT_SIZE'][0],
                                 frame_width=info['NETWORK_INPUT_SIZE'][1],
                                 agent_history_length=info['HISTORY_SIZE'],
                                 batch_size=info['BATCH_SIZE'],
                                 num_heads=info['N_ENSEMBLE'],
                                 bernoulli_probability=info['BERNOULLI_PROBABILITY'])

    random_state = np.random.RandomState(info["SEED"])
    action_getter = ActionGetter(n_actions=env.num_actions,
                                 eps_initial=info['EPS_INITIAL'],
                                 eps_final=info['EPS_FINAL'],
                                 eps_final_frame=info['EPS_FINAL_FRAME'],
                                 eps_annealing_frames=info['EPS_ANNEALING_FRAMES'],
                                 eps_evaluation=info['EPS_EVAL'],
                                 replay_memory_start_size=info['MIN_HISTORY_TO_LEARN'],
                                 max_steps=info['MAX_STEPS'])

    if args.model_loadpath != '':
        # load data from loadpath - save model load for later. we need some of
        # these parameters to setup other things
        print('loading model from: %s' %args.model_loadpath)
        model_dict = torch.load(args.model_loadpath)
        info = model_dict['info']
        info['DEVICE'] = device
        # set a new random seed
        info["SEED"] = model_dict['cnt']
        model_base_filedir = os.path.split(args.model_loadpath)[0]
        start_step_number = start_last_save = model_dict['cnt']
        info['loaded_from'] = args.model_loadpath
        perf = model_dict['perf']
        start_step_number = perf['steps'][-1]
    else:
        # create new project
        perf = {'steps':[],
                'avg_rewards':[],
                'episode_step':[],
                'episode_head':[],
                'eps_list':[],
                'episode_loss':[],
                'episode_reward':[],
                'episode_times':[],
                'episode_relative_times':[],
                'eval_rewards':[],
                #'eval_num_states':[],
                'eval_steps':[]}

        start_step_number = 0
        start_last_save = 0
        # make new directory for this run in the case that there is already a
        # project with this name
        run_num = 0
        model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
        while os.path.exists(model_base_filedir):
            run_num +=1
            model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
        os.makedirs(model_base_filedir)
        print("----------------------------------------------")
        print("starting NEW project: %s"%model_base_filedir)

    model_base_filepath = os.path.join(model_base_filedir, info['NAME'])
    write_info_file(info, model_base_filepath, start_step_number)
    heads = list(range(info['N_ENSEMBLE']))
    seed_everything(info["SEED"])

    policy_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
    target_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                      n_actions=env.num_actions,
                                      network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                      num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
    if info['PRIOR']:
        if 'PRETRAIN' in info['IMPROVEMENT'] and 'PRIOR' not in info['IMPROVEMENT']:
            prior_net = TanhGaussianPolicy(
                        obs_dim=info['NETWORK_INPUT_SIZE'],
                        action_dim=env.num_actions,
                        n_ensemble = info['N_ENSEMBLE'],
                        history_size = info['HISTORY_SIZE'],
                        device = info['DEVICE'],
                        if_dueling = False,)
            # prior_net = QNet(n_actions=env.num_actions,
            #             network_output_size=info['NETWORK_INPUT_SIZE'],
            #             num_channels=info['HISTORY_SIZE']+1, dueling=False).to(device)
            if 'LOAD' in info['IMPROVEMENT'] and os.path.exists(info['PRETRAIN_MODEL_PATH']):
                diayn_dict = torch.load(os.path.join(info['PRETRAIN_MODEL_PATH'], "best.pkl"))
                prior_net.get_network().load_state_dict(diayn_dict['policy_net_state_dict'])

        else:
            prior_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                    n_actions=env.num_actions,
                                    network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                    num_channels=info['HISTORY_SIZE'], dueling=False).to(info['DEVICE'])
            # prior_target_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
            #                         n_actions=env.num_actions,
            #                         network_output_size=info['NETWORK_INPUT_SIZE'][0],
            #                         num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
        if 'DISCRIMINATOR' in info['IMPROVEMENT']:
            discriminator = EnsembleNet(n_ensemble=1,
                                    n_actions=info['N_ENSEMBLE'],
                                    network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                    num_channels=info['HISTORY_SIZE'], dueling=False).to(info['DEVICE'])
            opt_discriminator = optim.Adam(discriminator.parameters(), lr=info['ADAM_LEARNING_RATE'])

            # print("using randomized prior")
            # policy_net = NetWithPrior(policy_net, prior_net, info['PRIOR_SCALE'])
            # target_net = NetWithPrior(target_net, prior_net, info['PRIOR_SCALE'])

    target_net.load_state_dict(policy_net.state_dict())

    # create optimizer
    #opt = optim.RMSprop(policy_net.parameters(),
    #                    lr=info["RMS_LEARNING_RATE"],
    #                    momentum=info["RMS_MOMENTUM"],
    #                    eps=info["RMS_EPSILON"],
    #                    centered=info["RMS_CENTERED"],
    #                    alpha=info["RMS_DECAY"])
    opt = optim.Adam(policy_net.parameters(), lr=info['ADAM_LEARNING_RATE'])

    kl_loss = nn.KLDivLoss()
    ce_loss = nn.CrossEntropyLoss()
    #eval_states = []
    if args.model_loadpath is not '':
        # what about random states - they will be wrong now???
        # TODO - what about target net update cnt
        target_net.load_state_dict(model_dict['target_net_state_dict'])
        policy_net.load_state_dict(model_dict['policy_net_state_dict'])
        opt.load_state_dict(model_dict['optimizer'])
        print("loaded model state_dicts")
        if args.buffer_loadpath == '':
            args.buffer_loadpath = args.model_loadpath.replace('.pkl', '_train_buffer.npz')
            print("auto loading buffer from:%s" %args.buffer_loadpath)
            try:
                replay_memory.load_buffer(args.buffer_loadpath)
            except Exception as e:
                print(e)
                print('not able to load from buffer: %s. exit() to continue with empty buffer' %args.buffer_loadpath)

    train(start_step_number, start_last_save)

