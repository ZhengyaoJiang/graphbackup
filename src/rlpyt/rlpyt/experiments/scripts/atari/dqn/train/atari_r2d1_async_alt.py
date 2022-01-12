
import sys

from src.rlpyt.rlpyt.utils.launching.affinity import affinity_from_code
from src.rlpyt.rlpyt.samplers.async_.alternating_sampler import AsyncAlternatingSampler
from src.rlpyt.rlpyt.samplers.async_.collectors import DbGpuResetCollector
from src.rlpyt.rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from src.rlpyt.rlpyt.algos.dqn.r2d1 import R2D1
from src.rlpyt.rlpyt.agents.dqn.atari.atari_r2d1_agent import AtariR2d1AlternatingAgent
from src.rlpyt.rlpyt.runners.async_rl import AsyncRlEval
from src.rlpyt.rlpyt.utils.logging.context import logger_context
from src.rlpyt.rlpyt.utils.launching.variant import load_variant, update_config

from src.rlpyt.rlpyt.experiments.configs.atari.dqn.atari_r2d1 import configs


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)
    config["eval_env"]["game"] = config["env"]["game"]

    sampler = AsyncAlternatingSampler(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=DbGpuResetCollector,
        TrajInfoCls=AtariTrajInfo,
        eval_env_kwargs=config["eval_env"],
        **config["sampler"]
    )
    algo = R2D1(optim_kwargs=config["optim"], **config["algo"])
    agent = AtariR2d1AlternatingAgent(model_kwargs=config["model"],
        **config["agent"])
    runner = AsyncRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = "async_alt_" + config["env"]["game"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
