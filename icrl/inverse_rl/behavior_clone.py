# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
# Implementation adapted from https://github.com/araffin/sbx
import argparse
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import random
import time
from dataclasses import dataclass
from distutils.util import strtobool
from functools import partial
from typing import Sequence
import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import orbax_utils
import orbax.checkpoint
import collections
# import pybullet_envs  # noqa
from jax.experimental import checkify # noqa
from flax.training.train_state import TrainState
import dejax

from icrl.env.cost_env import make_cost_env_no_xla
# Add progress bar if available
try:
    from tqdm.rich import tqdm
except ImportError:
    tqdm = None




def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ICRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--eval-freq", type=int, default=-1,
        help="evaluate the agent every `eval_freq` steps (if negative, no evaluation)")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
        help="number of episodes to use for evaluation")
    parser.add_argument("--n-eval-envs", type=int, default=5,
        help="number of environments for evaluation")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="BlockedHalfCheetah_backward",
        help="the id of the environment to be used")
    parser.add_argument("--expert-state-path", type=str, default="sacpolicy/Reacher-expert",
        help="the path of the expert state data")
    parser.add_argument("--expert-data", type=str, default="expert_data/BlockedHalfCheetah_backward",
        help="the name of the expert data file")
    parser.add_argument("--reg-mult", type=float, default=5,
        help="the reward alpha parameter")
    parser.add_argument("--reward-alpha", type=float, default=0.0,
        help="the reward alpha parameter")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=1024,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--n-critics", type=int, default=2,
        help="the number of critic networks")
    parser.add_argument("--policy-frequency", type=int, default=1,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=0.001,
        help="entropy regularization coefficient")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=False,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args



def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk



class Actor(nn.Module):
    action_dim: Sequence[int]
    n_units: int = 256
    log_std_min: float = -10
    log_std_max: float = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class RLTrainState(TrainState):
    target_params: flax.core.FrozenDict = None


@partial(jax.jit, static_argnames="actor")
def sample_action(
    actor: Actor,
    actor_state: TrainState,
    observations: jnp.ndarray,
    key: jax.random.KeyArray,
) -> jnp.array:
    key, subkey = jax.random.split(key, 2)
    mean, log_std = actor.apply(actor_state.params, observations)
    action_std = jnp.exp(log_std)
    gaussian_action = mean + action_std * jax.random.normal(subkey, shape=mean.shape)
    action = jnp.tanh(gaussian_action)
    return action, key



@partial(jax.jit, static_argnames="actor")
def select_action(actor: Actor, actor_state: TrainState, observations: jnp.ndarray) -> jnp.array:
    return actor.apply(actor_state.params, observations)[0]


def scale_action(action_space: gym.spaces.Box, action: np.ndarray) -> np.ndarray:
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)

    :param action: Action to scale
    :return: Scaled action
    """
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0


def unscale_action(action_space: gym.spaces.Box, scaled_action: np.ndarray) -> np.ndarray:
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)

    :param scaled_action: Action to un-scale
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


@dataclass
class SB3Adapter:
    """
    Adapter in order to use ``evaluate_policy()`` from Stable-Baselines3.
    """

    actor: Actor
    actor_state: RLTrainState
    key: jax.random.KeyArray
    action_space: gym.spaces.Box

    def predict(self, observations: np.ndarray, deterministic=True, state=None, episode_start=None):
        if deterministic:
            actions = select_action(self.actor, self.actor_state, observations)
        else:
            actions, self.key = sample_action(self.actor, self.actor_state, observations, self.key)

        # Clip due to numerical instability
        actions = np.clip(actions, -1, 1)
        # Rescale to proper domain when using squashing
        actions = unscale_action(self.action_space, actions)

        return actions, None


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


AgentState = collections.namedtuple(
    "AgentState",
    [
        "buffer_state", 
        "actor_state",
        "vf_state",
        "qf_state",
        "reward_qf_state",
        "reward_vf_state",
        "reward_lag", 
        "key"
    ]
)
def get_n_traj(expert_data):
    N_traj=3
    expert_obs=expert_data["traj_obs"].reshape(-1,100,11)[:,jnp.arange(N_traj)].reshape(-1,11)
    expert_action=expert_data["action"].reshape(-1,100,2)[:,jnp.arange(N_traj)].reshape(-1,2)
    expert_next_obs = expert_data["traj_next_obs"].reshape(-1,100,11)[:,jnp.arange(N_traj)].reshape(-1,11)
    expert_dones = expert_data["dones"].reshape(-1,100,1)[:,jnp.arange(N_traj)].reshape(-1)
    expert_action = jnp.array(expert_action).astype(jnp.float32)
    expert_obs = jnp.array(expert_obs).astype(jnp.float32)
    expert_next_obs = jnp.array(expert_next_obs).astype(jnp.float32)
    expert_dones = jnp.array(expert_dones).astype(jnp.float32)

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time()*10)}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    # Use a separate key, so running with/without eval doesn't affect the results
    eval_key = jax.random.PRNGKey(args.seed)
    envs, cost_function = make_cost_env_no_xla(args.env_id, 1, args.seed)

    # Create networks
    key, actor_key, qf_key, ent_key = jax.random.split(key, 4)

    obs = jnp.array([envs.observation_space.sample()]).astype(jnp.float32)
    action = jnp.array([envs.action_space.sample()]).astype(jnp.float32)


    actor = Actor(action_dim=np.prod(envs.action_space.shape))





    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    all_rewards=[]
    all_costs=[]
    for global_i in range(10):
        expert_data = checkpointer.restore(args.expert_data)
        # random_indice=np.random.randint(0, len(expert_data["traj_obs"]), size=50)
        # expert_data=jax.tree_map(lambda x:x[random_indice],expert_data)
        expert_action = jnp.array(expert_data["action"]).astype(jnp.float32)
        expert_obs = jnp.array(expert_data["traj_obs"]).astype(jnp.float32)
        expert_next_obs = jnp.array(expert_data["traj_next_obs"]).astype(jnp.float32)
        expert_dones = jnp.array(expert_data["dones"]).astype(jnp.float32)
        expert_rewards= jnp.array(expert_data["rewards"]).astype(jnp.float32)



        expert_actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, obs),
            tx=optax.adam(learning_rate=args.policy_lr),
        )
        def l2_loss(x):
            return (x ** 2).mean()
        def loss(params,obs,action):
            logits, stdlog = actor.apply(params, obs)
            var = jnp.exp(2 * stdlog)
            logprob = (
                -((action - logits) ** 2) / (2 * var)
                - stdlog
                - jnp.log(jnp.sqrt(2 * jnp.pi))
            ).sum()
            l2loss=50.0*sum(
            l2_loss(w) 
            for w in jax.tree_leaves(params["params"])
            )
            return -logprob+l2loss
        loss_grad_fn = jax.value_and_grad(loss)
        def train_agent(i,carry):
            expert_actor_state,key=carry
            key,sample_key = jax.random.split(key)
            random_indice=jax.random.randint(sample_key,(1024,), 0,len(expert_data["traj_obs"]))
            loss, grads = loss_grad_fn(
                expert_actor_state.params,
                expert_obs[random_indice],
                expert_action[random_indice])
            expert_actor_state = expert_actor_state.apply_gradients(grads=grads)
            return expert_actor_state,key
        @jax.jit
        def get_train_agent(expert_actor_state,key):
            expert_actor_state,key = jax.lax.fori_loop(
                0, 10000, train_agent, (expert_actor_state,key)
            )
            return expert_actor_state
        expert_actor_state=get_train_agent(expert_actor_state,key)

        reward_deque = collections.deque(maxlen=1000)
        cost_deque = collections.deque(maxlen=1000)
        length_deque = collections.deque(maxlen=1000)

        # TRY NOT TO MODIFY: start the game
        obs, info = envs.reset()
        obs = obs.astype(np.float32)
        # Display progress bar if available
        cum_reward = 0
        cum_cost = 0
        lengths = 0
        for global_step in range(5):
            for i in range(1000):
                actions, key = sample_action(actor, expert_actor_state, obs, key)
                input_actions = unscale_action(envs.action_space, actions.__array__())
                (next_obs, rewards, dones, truncated, infos) = envs.step(input_actions)
                costs = cost_function(next_obs, rewards, dones, truncated, infos)
                next_obs = next_obs.astype(np.float32)

                obs = next_obs
                cum_reward += rewards
                cum_cost += costs
                lengths += 1
                if dones.any() or truncated.any():
                    reward_deque.append(cum_reward)
                    cost_deque.append(cum_cost)
                    length_deque.append(lengths)
                    cum_reward = 0
                    cum_cost = 0
                    lengths = 0
                    (obs, rewards, dones, truncated, infos) = envs.step(input_actions)
                    obs = obs.astype(np.float32)
        all_rewards.append(np.array(reward_deque).mean())
        all_costs.append(np.array(cost_deque).mean())
    print(np.array(all_rewards).mean(),np.array(all_rewards).std())
    print(np.array(all_costs).mean(),np.array(all_costs).std())
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state=expert_actor_state
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(f"bcpolicy/{args.env_id}/{run_name}", state, save_args=save_args)
