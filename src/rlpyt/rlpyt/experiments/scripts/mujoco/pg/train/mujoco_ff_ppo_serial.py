
import sys

from src.rlpyt.rlpyt.utils.launching.affinity import affinity_from_code
from src.rlpyt.rlpyt.samplers.serial.sampler import SerialSampler
from src.rlpyt.rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from src.rlpyt.rlpyt.envs.gym import make as gym_make
from src.rlpyt.rlpyt.algos.pg.ppo import PPO
from src.rlpyt.rlpyt.agents.pg.mujoco import MujocoFfAgent
from src.rlpyt.rlpyt.runners.minibatch_rl import MinibatchRl
from src.rlpyt.rlpyt.utils.logging.context import logger_context
from src.rlpyt.rlpyt.utils.launching.variant import load_variant, update_config

from src.rlpyt.rlpyt.experiments.configs.mujoco.pg.mujoco_ppo import configs


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=config["env"],
        CollectorCls=CpuResetCollector,
        **config["sampler"]
    )
    algo = PPO(optim_kwargs=config["optim"], **config["algo"])
    agent = MujocoFfAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = "ppo_" + config["env"]["id"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
