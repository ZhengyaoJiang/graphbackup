
import sys

from src.rlpyt.rlpyt.utils.launching.affinity import affinity_from_code
from src.rlpyt.rlpyt.samplers.gpu.sampler import GpuSampler
from src.rlpyt.rlpyt.samplers.gpu.collectors import GpuWaitResetCollector
from src.rlpyt.rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from src.rlpyt.rlpyt.algos.pg.ppo import PPO
from src.rlpyt.rlpyt.agents.pg.atari import AtariLstmAgent
from src.rlpyt.rlpyt.runners.minibatch_rl import MinibatchRl
from src.rlpyt.rlpyt.utils.logging.context import logger_context
from src.rlpyt.rlpyt.utils.launching.variant import load_variant, update_config

from src.rlpyt.rlpyt.experiments.configs.atari.pg.atari_lstm_ppo import configs


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    sampler = GpuSampler(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=GpuWaitResetCollector,
        TrajInfoCls=AtariTrajInfo,
        **config["sampler"]
    )
    algo = PPO(optim_kwargs=config["optim"], **config["algo"])
    agent = AtariLstmAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = config["env"]["game"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
