import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

from dqn_model import EnsembleNet, NetWithPrior
from dqn_utils import seed_everything, write_info_file, generate_gif, save_checkpoint
from env import Environment
from replay import ReplayMemory
from diayn_policies import *
import config

print("Using the torch version : ", torch.__version__)

EPS = 1E-6


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

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


class DIAYN():
    def __init__(
            self,
            env,
            discount=0.99,
            reward_scale=1.0,

            policy_lr=3E-4, #1e-3,
            qf_lr=3E-4 #1e-3,
            value_lr=1e-3,
            discriminator_lr=1e-3,
            optimizer_class=optim.Adam,

            lr=3E-3,
            scale_entropy=1,
            tau=0.01,
            num_skills=info['N_ENSEMBLE'],
            save_full_state=False,
            find_best_skill_interval=10,
            best_skill_n_rollouts=10,
            learn_p_z=False,
            include_actions=False,
            add_p_z=True,

            soft_target_tau=5e-3, #1e-2,
            target_update_period=1,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            train_policy_with_reparameterization=True,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            policy_update_period=1,
            n_train_steps_total = 0
    ):
        self.env = env
        self.replay_memory = ReplayMemory(size=info['BUFFER_SIZE'],
                                 frame_height=info['NETWORK_INPUT_SIZE'][0],
                                 frame_width=info['NETWORK_INPUT_SIZE'][1],
                                 agent_history_length=info['HISTORY_SIZE'],
                                 batch_size=info['BATCH_SIZE'],
                                 num_heads=info['N_ENSEMBLE'],
                                 bernoulli_probability=info['BERNOULLI_PROBABILITY'])

        self.policy = TanhGaussianPolicy(
                                obs_dim=info['NETWORK_INPUT_SIZE'],
                                action_dim=env.num_actions,
                                if_dueling = info['DUELING'],)

        self.qf1 = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                n_actions=env.num_actions,
                                network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
        self.qf2 = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                n_actions=env.num_actions,
                                network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
        # self.target_qf1 = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
        #                         n_actions=env.num_actions,
        #                         network_output_size=info['NETWORK_INPUT_SIZE'][0],
        #                         num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
        # self.target_qf2 = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
        #                         n_actions=env.num_actions,
        #                         network_output_size=info['NETWORK_INPUT_SIZE'][0],
        #                         num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
        self.discriminator = = EnsembleNet(n_ensemble=1,
                                n_actions=info['N_ENSEMBLE'],
                                network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                num_channels=info['HISTORY_SIZE'], dueling=False).to(info['DEVICE'])
        self.value_network = EnsembleNet(n_ensemble=1,
                                n_actions=1,
                                network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])

        # self.higher_level_policy = EnsembleNet(n_ensemble=1,
        #                         n_actions=info['N_ENSEMBLE'],
        #                         network_output_size=info['NETWORK_INPUT_SIZE'][0],
        #                         num_channels=info['HISTORY_SIZE'], dueling=False).to(info['DEVICE'])

        self.target_vf = EnsembleNet(n_ensemble=1,
                                n_actions=1,
                                network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])

        self.train_policy_with_reparameterization = train_policy_with_reparameterization

        self.soft_target_tau = soft_target_tau
        self.policy_update_period = policy_update_period
        self.target_update_period = target_update_period
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.train_policy_with_reparameterization = (
            train_policy_with_reparameterization
        )

        self.tau = tau
        self.num_skills = num_skills
        self.save_full_state = save_full_state
        self.find_best_skill_interval = find_best_skill_interval
        self.best_skill_n_rollouts = best_skill_n_rollouts
        self.learn_p_z = learn_p_z
        self.include_actions = include_actions
        self.add_p_z = add_p_z
        self.random_state = np.random.RandomState(info["SEED"])
        self.heads = list(range(info['N_ENSEMBLE']))

        self.p_z = np.full(num_skills, 1.0 / num_skills)

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discriminator_optimizer = optimizer_class(
            self.discriminator.parameters(),
            lr=discriminator_lr
        )

        self.vf_optimizer = optimizer_class(
            self.value_network.parameters(),
            lr=value_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale

        self.action_dim = self.env.action_space
        self.obs_dim = info['NETWORK_INPUT_SIZE']
        self._n_train_steps_total = n_train_steps_total

    def sample_empowerment_latents(self, observation):
        """Samples z from p(z), using probabilities in self.p_z."""
        self.random_state.shuffle(self.heads)
        active_head = self.heads[0]
        return active_head #self.higher_policy(observation).sample()

    # def split_obs(self, obs):
    #     obs, z_one_hot = obs[:self.obs_dim], obs[self.obs_dim:]
    #     return obs, z_one_hot

    def update_critic(self, observation, action,
                      p_z_given, next_observation, active_head, done):

        """
        Create minimization operation for the critic Q function.
        :return: TD Loss, Empowerment Reward
        """

        # Get the q value for the observation(obs, z_one_hot) and action.

        q_value_1 = self.qf1(observation, active_head)[action]
        q_value_2 = self.qf2(observation, active_head)[action]

        if self.include_actions:
            logits = self.discriminator(observation, action)
        else:
            logits = self.discriminator(observation)

        # The empowerment reward is defined as the cross entropy loss between the
        # true skill and the selected skill.
        empowerment_reward = -1 * torch.nn.CrossEntropyLoss()(z_one_hot, logits)

        p_z = torch.sum(p_z_given*z_one_hot, axis=1)
        log_p_z = torch.log(p_z+EPS)

        if self.add_p_z:
            empowerment_reward -= log_p_z

        # Now we will calculate the value function and critic Q function update.
        vf_target_next_obs = self.target_vf(next_observation)
        v_pred = self.value_network(observation)

        # Calculate the targets for the Q function (Calculate Q Function Loss)
        q_target = self.reward_scale*empowerment_reward + (1 - done) * self.discount * vf_target_next_obs
        qf1_loss = self.qf_criterion(q_value_1, q_target.detach())
        qf2_loss = self.qf_criterion(q_value_2, q_target.detach())

        return qf1_loss, qf2_loss, empowerment_reward, v_pred

    def update_state_value(self, observation, action, p_z_given,
                           next_observation, active_head,
                           done):
        """
        Creates minimization operations for the state value functions.

        In principle, there is no need for a separate state value function
        approximator, since it could be evaluated using the Q-function and
        policy. However, in practice, the separate function approximator
        stabilizes training.

        :return:
        """

        qf1_loss, qf2_loss, empowerment_reward, _ = self.update_critic(observation=observation,
                                                                       action=action,
                                                                       p_z_given=p_z_given,
                                                                       next_observation=next_observation,
                                                                       active_head=active_head,
                                                                       done=done)

        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.policy(observation, skill=active_head,
                                     reparameterize=self.train_policy_with_reparameterization,
                                     return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        q_new_actions = torch.min(
            self.qf1(observation, new_actions),
            self.qf2(observation, new_actions),
        )

        v_pred = self.value_network(observation)

        """
               Alpha Loss (if applicable)
               """
        if self.use_automatic_entropy_tuning:
            """
            Alpha Loss
            """
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
            alpha_loss = 0

        policy_distribution = self.policy.get_distibution(observation)
        log_pi = policy_distribution.log_pi

        v_target = q_new_actions - alpha * log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        policy_loss = None
        if self._n_train_steps_total % self.policy_update_period == 0:
            """
            Policy Loss
            """
            if self.train_policy_with_reparameterization:
                policy_loss = (alpha * log_pi - q_new_actions).mean()
            else:
                log_policy_target = q_new_actions - v_pred
                policy_loss = (
                        log_pi * (alpha * log_pi - log_policy_target).detach()
                ).mean()
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
            pre_tanh_value = policy_outputs[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value ** 2).sum(dim=1).mean()
            )
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            policy_loss = policy_loss + policy_reg_loss

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        if self._n_train_steps_total % self.target_update_period == 0:
            soft_update_from_to(
                self.value_network, self.target_vf, self.soft_target_tau
            )

        """
        Save some statistics for eval using just one batch.
        """
        # if self.need_to_update_eval_statistics:
        #     self.need_to_update_eval_statistics = False
        #     if policy_loss is None:
        #         if self.train_policy_with_reparameterization:
        #             policy_loss = (log_pi - q_new_actions).mean()
        #         else:
        #             log_policy_target = q_new_actions - v_pred
        #             policy_loss = (
        #                 log_pi * (log_pi - log_policy_target).detach()
        #             ).mean()

        #         mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        #         std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        #         pre_tanh_value = policy_outputs[-1]
        #         pre_activation_reg_loss = self.policy_pre_activation_weight * (
        #             (pre_tanh_value**2).sum(dim=1).mean()
        #         )
        #         policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        #         policy_loss = policy_loss + policy_reg_loss

        #     self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
        #     self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
        #     self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
        #     self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
        #         policy_loss
        #     ))

        #     if self.use_automatic_entropy_tuning:
        #         self.eval_statistics['Alpha'] = alpha.item()
        #         self.eval_statistics['Alpha Loss'] = alpha_loss.item()

        return vf_loss, alpha_loss, alpha, qf1_loss, qf2_loss, empowerment_reward, policy_loss

    def update_discriminator(self, observation, action):
        """

        Creates the minimization operation for the discriminator.

        :return:
        """
        if self.include_actions:
            logits = self.discriminator(observation, action)
        else:
            logits = self.discriminator(observation)

        discriminator_loss = torch.nn.CrossEntropyLoss()(logits, z_one_hot)

        """
        Update the discriminator
        """

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        return discriminator_loss

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

        state = torch.Tensor(state.astype(np.float)/info['NORM_BY'])[None,:].to(info['DEVICE'])
        if not evaluation:
            self.higher_level_policy.eval()
            logits = self.higher_level_policy(states, 0)
            active_heads = OneHotCategorical(logits=logits)
            active_head = active_heads.sample()
        a, vals = self.policy.get_action(state, active_head)
        return active_head, a

    def batch_upadte(self, rewards, terminals, obs, actions, next_obs, active_head):
        p_z = self.sample_empowerment_latents(obs)

        """

        Update the networks

        """

        rews = np.mean(ptu.get_numpy(rewards))

        vf_loss, alpha_loss, alpha, qf1_loss, qf2_loss, emp_reward, pol_loss = self.update_state_value(
            observation=obs,
            action=actions,
            done=terminals,
            next_observation=next_obs,
            active_head= active_head
            p_z_given=p_z
        )

        # Update the discriminator
        discriminator_loss = self.update_discriminator(observation=obs,
                                                       action=actions)

        i = self._n_train_steps_total


        self._n_train_steps_total += 1
        return np.mean(pol_loss)

    def train(self, step_number=0, last_save=0):
        """Contains the training and evaluation loops"""
        epoch_num = len(perf['steps'])
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
                epoch_num += 1
                ptloss_list = []
                while not terminal:
                    active_head, action = self.pt_get_action(step_number, state=state, active_head=None)
                    if life_lost:
                        action = 1
                    next_state, reward, life_lost, terminal = env.step(action)
                    # Store transition in the replay memory
                    self.replay_memory.add_experience(action=action,
                                                    frame=next_state[-1],
                                                    reward=np.sign(reward), # TODO -maybe there should be +1 here
                                                    terminal=life_lost,
                                                    active_head= active_head)

                    step_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward
                    state = next_state

                    if step_number % info['LEARN_EVERY_STEPS'] == 0 and step_number > info['MIN_HISTORY_TO_LEARN']:
                        _states, _actions, _rewards, _next_states, _terminal_flags, _active_heads, _masks = self.replay_memory.get_minibatch(info['BATCH_SIZE'])
                        ptloss = self.batch_upadte(_rewards, _terminal_flags, _states, _actions, _next_states)
                        ptloss_list.append(ptloss)
                    # if step_number % info['TARGET_UPDATE'] == 0 and step_number >  info['MIN_HISTORY_TO_LEARN']:
                    #     print("++++++++++++++++++++++++++++++++++++++++++++++++")
                    #     print('updating target network at %s'%step_number)
                    #     target_net.load_state_dict(policy_net.state_dict())
                    #     #prior_target_net.load_state_dict(prior_net.state_dict())

                last_save = handle_checkpoint(last_save, step_number)
            avg_eval_reward = evaluate(step_number)

    def evaluate(self, step_number):
        print("""
            #########################
            ####### Evaluation ######
            #########################
            """)
        eval_rewards = []
        evaluate_step_number = 0
        frames_for_gif = []
        results_for_eval = []
        # only run one
        for i in range(1):
            state = env.reset()
            episode_reward_sum = 0
            terminal = False
            life_lost = True
            episode_steps = 0
            while not terminal:
                if life_lost:
                    action = 1
                else:
                    active_head, action = self.pt_get_action(step_number, state, active_head=0, evaluation=True)
                next_state, reward, life_lost, terminal = env.step(action)
                evaluate_step_number += 1
                episode_steps +=1
                episode_reward_sum += reward
                if not i:
                    # only save first episode
                    frames_for_gif.append(env.ale.getScreenRGB())
                    results_for_eval.append("%s, %s, %s, %s" %(action, reward, life_lost, terminal))
                if not episode_steps%100:
                    print('eval', episode_steps, episode_reward_sum)
                state = next_state
            eval_rewards.append(episode_reward_sum)

        print("Evaluation score:\n", np.mean(eval_rewards))
        generate_gif(model_base_filedir, step_number, frames_for_gif, eval_rewards[0], name='test', results=results_for_eval)
            efile = os.path.join(model_base_filedir, 'eval_rewards.txt')
        with open(efile, 'a') as eval_reward_file:
            print(step_number, np.mean(eval_rewards), file=eval_reward_file)
        return np.mean(eval_rewards)

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_vf,
            self.discriminator,
            self.value_network
        ]

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("running on %s"%device)

    info = {
        "GAME":'roms/freeway.bin', # gym prefix
        "DEVICE":device, #cpu vs gpu set by argument
        "model_path":'diayn_net', # start files with name
        "TARGET_UPDATE":10000, # how often to update target network
        "MIN_HISTORY_TO_LEARN":0, # in environment frames
        "NORM_BY":255.,  # divide the float(of uint) by this number to normalize - max val of data is 255
        "MAX_STEPS":int(50e3), # 50e6 steps is 200e6 frames
        "MAX_EPISODE_STEPS":27000, # Orig dqn give 18k steps, Rainbow seems to give 27k steps
        "FRAME_SKIP":4, # deterministic frame skips to match deepmind
        "MAX_NO_OP_FRAMES":30, # random number of noops applied to beginning of each episode
        "DEAD_AS_END":True, # do you send finished=true to agent while training when it loses a life
        "NETWORK_INPUT_SIZE":(84,84),
        "HISTORY_SIZE":4, # how many past frames to use for state input
        "BUFFER_SIZE":int(1e3), # Buffer size for experience replay
        'EVAL_FREQUENCY': 2500,
        'CHECKPOINT_EVERY_STEPS':50000,
        "DUELING":True, # use dueling dqn
        "N_ENSEMBLE": 9,
        "LEARN_EVERY_STEPS":4,
        "SEED":101,

    }
    seed_everything(info["SEED"])
    diayn = DIAYN()
    diayn.train()
