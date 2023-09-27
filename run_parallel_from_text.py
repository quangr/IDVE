import itertools
import multiprocessing
import shlex
from multiprocessing import Queue
import argparse
import random
import time


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument(
    "--available_gpus",
    type=int,
    default=[0,1,2,3],
    nargs='+',
    help="List of available GPUs to use",
)
parser.add_argument(
    "--filename",
    type=str,
    help="Number of GPUs to use",
)
parser.add_argument(
    "--dry", action=argparse.BooleanOptionalAction, default=False, help="dry run"
)


fileparser = argparse.ArgumentParser(allow_abbrev=False)
fileparser.add_argument(
    "--filename",
    type=str,
    help="Number of GPUs to use",
)

def parse_user_defined_args(unknown_args):
    """Parses the user-defined arguments and returns a dictionary of the arguments and their values."""
    parsed_args = {}
    current_arg = None
    for arg in unknown_args:
        if arg.startswith('--'):
            # If the current argument starts with '-', it's a new user-defined argument
            current_arg = arg.lstrip('--')
            parsed_args[current_arg] = []
        else:
            # Otherwise, it's a value for the current user-defined argument
            if current_arg:
                parsed_args[current_arg].append(arg)
    return parsed_args
def product_dict(**kwargs):
  """
  Returns a list of dictionaries, where each dictionary is a product of the values
  in the input dictionary.

  Args:
    **kwargs: A dictionary of values.

  Returns:
    A list of dictionaries.
  """

  keys = kwargs.keys()
  for instance in itertools.product(*kwargs.values()):
    yield dict(zip(keys, instance))


run_list=[]
with open('mujoco_exp.txt', 'r') as f:
    for line in f:
        args = shlex.split(line)
        args, unknown_args = fileparser.parse_known_args(args)

        # Parse the unknown arguments using the custom function
        parsed_args = parse_user_defined_args(unknown_args)
        combinations = product_dict(**parsed_args)
        print(parsed_args)
        run_list+=[(args.filename,y) for y in combinations]
        # Initialize the GPU queue with IDs of available GPUs

args = parser.parse_args()

batch_size = args.batch_size
available_gpus = args.available_gpus
gpu_num=len(available_gpus)
worker_count = gpu_num * batch_size

gpu_queue = Queue()


def dry_run_a_py(filename, **args):
    additional_args = " ".join([f"--{key} {value}" for key, value in args.items()])
    command = f"python {filename} {additional_args}"
    print("Dry Run Command:", command)

def run_a_py(filename, **args):
    # Acquire a GPU from the queue
    gpu_id = gpu_queue.get()
    try:
        sleep_time = random.uniform(0, 1)
        time.sleep(sleep_time)
        additional_args = " ".join([f"--{key}={value}" for key, value in args.items()])
        command = f"python {filename} {additional_args}"
        import subprocess, os

        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        my_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(1/batch_size*0.8)
        subprocess.run(shlex.split(command), check=True, env=my_env)
    finally:
        # Release the GPU back to the queue
        gpu_queue.put(gpu_id)


def starmap_with_kwargs(pool, fn, carry):
    args_for_starmap =[(fn,filename, kwargs) for (filename, kwargs) in carry]
    pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, filename, kwargs):
    return fn(filename,**kwargs)


for index in available_gpus:
    for _ in range(batch_size):
        gpu_queue.put(index)

# Use dry run if specified, otherwise run each combination in parallel using multiprocessing
print(run_list)
if args.dry:
    with multiprocessing.Pool(1) as pool:
        starmap_with_kwargs(pool,dry_run_a_py, run_list)
else:
    with multiprocessing.Pool(worker_count) as pool:
        # Use starmap to submit each combination to the worker pool
        starmap_with_kwargs(pool,run_a_py, run_list)
