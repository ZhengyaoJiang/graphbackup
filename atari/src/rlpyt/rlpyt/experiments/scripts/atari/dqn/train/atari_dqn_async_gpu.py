
import sys

from src.rlpyt.rlpyt.utils.launching.affinity import affinity_from_code
from src.rlpyt.rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler
from src.rlpyt.rlpyt.samplers.async_.collectors import DbGpuResetCollector
from src.rlpyt.rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from src.rlpyt.rlpyt.algos.dqn.dqn import DQN
from src.rlpyt.rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from src.rlpyt.rlpyt.runners.async_rl import AsyncRlEval
from src.rlpyt.rlpyt.utils.logging.context import logger_context
from src.rlpyt.rlpyt.utils.launching.variant import load_variant, update_config

from src.rlpyt.rlpyt.experiments.configs.atari.dqn.atari_dqn import configs


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)
    config["eval_env"]["game"] = config["env"]["game"]

    sampler = AsyncGpuSampler(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=DbGpuResetCollector,
        TrajInfoCls=AtariTrajInfo,
        eval_env_kwargs=config["eval_env"],
        **config["sampler"]
    )
    algo = DQN(optim_kwargs=config["optim"], **config["algo"])
    agent = AtariDqnAgent(model_kwargs=config["model"], **config["agent"])
    runner = AsyncRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = "async_gpu_" + config["env"]["game"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
