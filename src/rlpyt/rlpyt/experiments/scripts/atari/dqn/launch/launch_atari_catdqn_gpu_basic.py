
from src.rlpyt.rlpyt.utils.launching.affinity import encode_affinity
from src.rlpyt.rlpyt.utils.launching.exp_launcher import run_experiments
from src.rlpyt.rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/atari/dqn/train/atari_catdqn_gpu.py"
affinity_code = encode_affinity(
    n_cpu_core=4,
    n_gpu=1,
    hyperthread_offset=8,
    n_socket=1,
    # cpu_per_run=2,
)
runs_per_setting = 2
experiment_title = "atari_catdqn_basic"
variant_levels = list()

games = ["pong", "seaquest", "qbert", "chopper_command"]
values = list(zip(games))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "game")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

default_config_key = "catdqn"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
