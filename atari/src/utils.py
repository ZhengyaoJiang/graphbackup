from src.rlpyt.rlpyt.experiments.configs.atari.dqn.atari_dqn import configs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class dummy_context_mgr:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def set_config(args, game):
    # TODO: Use Hydra to manage configs
    config = configs['ernbw']
    config['env']['game'] = game
    config["env"]["grayscale"] = args.grayscale
    config["env"]["num_img_obs"] = args.framestack
    config["eval_env"]["game"] = config["env"]["game"]
    config["eval_env"]["grayscale"] = args.grayscale
    config["eval_env"]["num_img_obs"] = args.framestack
    config['env']['imagesize'] = args.imagesize
    config['eval_env']['imagesize'] = args.imagesize
    config['env']['seed'] = args.seed
    config['eval_env']['seed'] = args.seed
    config["model"]["dueling"] = bool(args.dueling)
    config["algo"]["min_steps_learn"] = args.min_steps_learn
    config["algo"]["n_step_return"] = args.n_step
    config["algo"]["batch_size"] = args.batch_size
    config["algo"]["learning_rate"] = args.learning_rate
    config['algo']['replay_ratio'] = args.replay_ratio
    config['algo']['target_update_interval'] = args.target_update_interval
    config['algo']['target_update_tau'] = args.target_update_tau
    config['algo']['eps_steps'] = args.eps_steps
    config["algo"]["clip_grad_norm"] = args.max_grad_norm
    config['algo']['pri_alpha'] = 0.5
    config['algo']['discount'] = args.discount
    config['algo']['pri_beta_steps'] = int(10e4)
    config['algo']['double_dqn'] = args.double
    config['optim']['eps'] = 0.00015
    config["sampler"]["eval_max_trajectories"] = 100
    config["sampler"]["eval_n_envs"] = 100
    config["sampler"]["eval_max_steps"] = 100*28000  # 28k is just a safe ceiling
    config['sampler']['batch_B'] = args.batch_b
    config['sampler']['batch_T'] = args.batch_t

    config['agent']['eps_init'] = args.eps_init
    config['agent']['eps_final'] = args.eps_final
    config["model"]["noisy_nets_std"] = args.noisy_nets_std
    config["model"]["architecture"] = args.architecture

    if args.noisy_nets:
        config['agent']['eps_eval'] = 0.001

    # New SPR Arguments
    config["model"]["imagesize"] = args.imagesize
    config["model"]["jumps"] = args.jumps
    config["model"]["dynamics_blocks"] = args.dynamics_blocks
    config["model"]["spr"] = args.spr
    config["model"]["noisy_nets"] = args.noisy_nets
    config["model"]["momentum_encoder"] = args.momentum_encoder
    config["model"]["shared_encoder"] = args.shared_encoder
    config["model"]["local_spr"] = args.local_spr
    config["model"]["global_spr"] = args.global_spr
    config["model"]["distributional"] = args.distributional
    config["model"]["renormalize"] = args.renormalize
    config["model"]["norm_type"] = args.norm_type
    config["model"]["augmentation"] = args.augmentation
    config["model"]["q_l1_type"] = args.q_l1_type
    config["model"]["dropout"] = args.dropout
    config["model"]["time_offset"] = args.time_offset
    config["model"]["aug_prob"] = args.aug_prob
    config["model"]["target_augmentation"] = args.target_augmentation
    config["model"]["eval_augmentation"] = args.eval_augmentation
    config["model"]["classifier"] = args.classifier
    config["model"]["final_classifier"] = args.final_classifier
    config['model']['momentum_tau'] = args.momentum_tau
    config["model"]["dqn_hidden_size"] = args.dqn_hidden_size
    config["model"]["model_rl"] = args.model_rl_weight
    config["model"]["residual_tm"] = args.residual_tm
    config["algo"]["model_rl_weight"] = args.model_rl_weight
    config["algo"]["reward_loss_weight"] = args.reward_loss_weight
    config["algo"]["model_spr_weight"] = args.model_spr_weight
    config["algo"]["t0_spr_loss_weight"] = args.t0_spr_loss_weight
    config["algo"]["time_offset"] = args.time_offset
    config["algo"]["distributional"] = args.distributional
    config["algo"]["double"] = args.double
    config["algo"]["delta_clip"] = args.delta_clip
    config["algo"]["prioritized_replay"] = args.prioritized_replay
    config["algo"]["backup"] = args.backup

    return config


import collections
import timeit
import numpy as np

class Timings:
    """Not thread-safe."""

    def __init__(self):
        self._means = collections.defaultdict(int)
        self._vars = collections.defaultdict(int)
        self._counts = collections.defaultdict(int)
        self.reset()

    def reset(self):
        self.last_time = timeit.default_timer()

    def time(self, name):
        """Save an update for event `name`.

        Nerd alarm: We could just store a
            collections.defaultdict(list)
        and compute means and standard deviations at the end. But thanks to the
        clever math in Sutton-Barto
        (http://www.incompleteideas.net/book/first/ebook/node19.html) and
        https://math.stackexchange.com/a/103025/5051 we can update both the
        means and the stds online. O(1) FTW!
        """
        now = timeit.default_timer()
        x = now - self.last_time
        self.last_time = now

        n = self._counts[name]

        mean = self._means[name] + (x - self._means[name]) / (n + 1)
        var = (
            n * self._vars[name] + n * (self._means[name] - mean) ** 2 + (x - mean) ** 2
        ) / (n + 1)

        self._means[name] = mean
        self._vars[name] = var
        self._counts[name] += 1

    def means(self):
        return self._means

    def vars(self):
        return self._vars

    def stds(self):
        return {k: v ** 0.5 for k, v in self._vars.items()}

    def summary2(self, prefix=""):
        """
        used for uneven count case
        """
        means = self.means()
        counts = np.array([self._counts[k] for k in means.keys()])
        total = sum(np.array(list(means.values()))*counts)

        result = prefix
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n    %s: %.6fs (%.2f%%) " % (
                k,
                self._counts[k] * means[k],
                100 * self._counts[k] * means[k] / total,
            )
        result += "\nTotal: %.6fs" % (total)
        return result

    def summary(self, prefix=""):
        means = self.means()
        stds = self.stds()
        total = sum(means.values())

        result = prefix
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n    %s: %.6fms +- %.6fms (%.2f%%) " % (
                k,
                1000 * means[k],
                1000 * stds[k],
                100 * means[k] / total,
            )
        result += "\nTotal: %.6fms" % (1000 * total)
        return result