
from src.rlpyt.rlpyt.utils.launching.affinity import encode_affinity
from src.rlpyt.rlpyt.utils.launching.exp_launcher import run_experiments
from src.rlpyt.rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/atari/pg/train/atari_lstm_a2c_cpu.py"
# default_config_key = "0"
affinity_code = encode_affinity(  # Let it be kwargs?
    n_cpu_core=6,
    n_gpu=2,
    hyperthread_offset=8,
    n_socket=1,
    # cpu_per_run=4,
)
runs_per_setting = 2
experiment_title = "lstm_test_gpu"
variant_levels = list()

# learning_rate = [7e-4] * 4
# entropy_loss_coeff = [0.01, 0.02, 0.04, 0.08]
# values = list(zip(learning_rate, entropy_loss_coeff))
# dir_names = ["test_{}lr_{}ent".format(*v) for v in values]
# keys = [("algo", "learning_rate"), ("algo", "entropy_loss_coeff")]
# variant_levels.append(VariantLevel(keys, values, dir_names))


learning_rate = [7e-4] * 2
batch_T = [5, 20]
values = list(zip(learning_rate, batch_T))
dir_names = ["test_{}lr_{}T".format(*v) for v in values]
keys = [("algo", "learning_rate"), ("sampler", "batch_T")]
variant_levels.append(VariantLevel(keys, values, dir_names))


games = ["pong", "seaquest"]
values = list(zip(games))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "game")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

default_config_key = "0"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
