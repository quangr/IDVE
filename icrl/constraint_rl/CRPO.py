# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
# Implementation adapted from https://github.com/araffin/sbx
import argparse
import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
import random
import time
from dataclasses import dataclass
from distutils.util import strtobool
from functools import partial
from typing import Sequence
from jax.experimental import checkify
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
from flax.training.train_state import TrainState
import dejax
from icrl.common.venv_wrappers import (
    MojocoEnvDtypeAct,
    VectorEnvClipAct,
)
from jax.config import config

from icrl.env.cost_env import make_cost_env, step_env_wrappeed_factory

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
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
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
    parser.add_argument("--env-id", type=str, default="Reacher-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--n-critics", type=int, default=2,
        help="the number of critic networks")
    parser.add_argument("--policy-frequency", type=int, default=1,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="entropy regularization coefficient")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
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


class Critic(nn.Module):
    n_units: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([x, action], -1)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class VectorCritic(nn.Module):
    n_units: int = 256
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            n_units=self.n_units,
        )(obs, action)
        return q_values


class Actor(nn.Module):
    action_dim: Sequence[int]
    n_units: int = 256
    log_std_min: float = -20
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


@jax.jit
def sample_action_and_log_prob(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    subkey: jax.random.KeyArray,
):
    action_std = jnp.exp(log_std)
    gaussian_action = mean + action_std * jax.random.normal(subkey, shape=mean.shape,dtype=jnp.float32)
    log_prob = -0.5 * ((gaussian_action - mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - log_std
    log_prob = log_prob.sum(axis=1)
    action = jnp.tanh(gaussian_action)
    log_prob -= jnp.sum(jnp.log((1 - action**2) + 1e-6), 1)
    return action, log_prob


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


@jax.jit
def soft_update(tau: float, qf_state: RLTrainState) -> RLTrainState:
    qf_state = qf_state.replace(target_params=optax.incremental_update(qf_state.params, qf_state.target_params, tau))
    return qf_state

@jax.jit
@checkify.checkify
def train_step(buffer_state, qf_state, actor_state, ent_coef_state,key):
    sample_key, key = jax.random.split(key, 2)
    observations,next_observations,actions,rewards,costs,dones=buffer.sample_fn(buffer_state, sample_key, args.batch_size)
    if args.autotune:
        ent_coef_value = ent_coef.apply({"params": ent_coef_state.params})

    qf_state, (qf_loss_value, qf_values), key = update_critic(
                actor_state,
                qf_state,
                ent_coef_value,
                observations,
                actions,
                next_observations,
                rewards,
                dones,
                key,
            )

    (actor_state, qf_state, actor_loss_value, key, entropy) = update_actor(
                actor_state,
                qf_state,
                ent_coef_value,
                observations,
                key,
            )
    qf_state = soft_update(args.tau, qf_state)

    if args.autotune:
        ent_coef_state, ent_coef_loss = update_temperature(ent_coef_state, entropy)
    return actor_state,qf_state,ent_coef_state,ent_coef_value,qf_loss_value,qf_values,actor_loss_value,ent_coef_loss,key
if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    def get_wrappers(envs):
        wrappers = [
            VectorEnvClipAct(envs.action_space.low, envs.action_space.high),
        ]
        if envs.action_space.dtype == np.float64:
            config.update("jax_enable_x64", True)
            wrappers=wrappers + [MojocoEnvDtypeAct()]
        return wrappers
    # wrappers=[]
    envs, handle, step_env, episode_stats,cost_function =make_cost_env(args.env_id,1,args.seed,get_wrappers)
    step_env_wrappeed=step_env_wrappeed_factory(step_env,cost_function)

    # Create networks
    key, actor_key, qf_key, ent_key = jax.random.split(key, 4)

    obs = jnp.array([envs.observation_space.sample()]).astype(jnp.float32)
    action = jnp.array([envs.action_space.sample()]).astype(jnp.float32)

    buffer = dejax.uniform_replay(max_size=args.buffer_size)
    buffer_state = buffer.init_fn((obs[0],obs[0],action[0],jnp.array([0.],dtype=jnp.float32),jnp.array([0.],dtype=jnp.float32),jnp.array([True])))

    actor = Actor(action_dim=np.prod(envs.action_space.shape))

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.policy_lr),
    )

    qf = VectorCritic(n_critics=args.n_critics)

    qf_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init({"params": qf_key}, obs, action),
        target_params=qf.init({"params": qf_key}, obs, action),
        tx=optax.adam(learning_rate=args.q_lr),
    )

    # Automatic entropy tuning
    if args.autotune:
        ent_coef = EntropyCoef(ent_coef_init=1.0)
        target_entropy = -np.prod(envs.action_space.shape).astype(np.float32)
        ent_coef_state = TrainState.create(
            apply_fn=ent_coef.apply,
            params=ent_coef.init(ent_key)["params"],
            tx=optax.adam(learning_rate=args.q_lr),
        )

    else:
        ent_coef_value = jnp.array(args.alpha)

    # Define update functions here to limit the need for static argname
    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf_state: RLTrainState,
        ent_coef_value: jnp.ndarray,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        key: jax.random.KeyArray,
    ):
        key, subkey = jax.random.split(key, 2)
        mean, log_std = actor.apply(actor_state.params, next_observations)
        next_state_actions, next_log_prob = sample_action_and_log_prob(mean, log_std, subkey)

        qf_next_values = qf.apply(qf_state.target_params, next_observations, next_state_actions)
        next_q_values = jnp.min(qf_next_values, axis=0)
        # td error + entropy term
        next_q_values = next_q_values - ent_coef_value * next_log_prob.reshape(-1, 1)
        # shape is (batch_size, 1)
        target_q_values = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * args.gamma * next_q_values

        def mse_loss(params):
            # shape is (n_critics, batch_size, 1)
            current_q_values = qf.apply(params, observations, actions)
            # mean over the batch and then sum for each critic
            critic_loss = 0.5 * ((target_q_values - current_q_values) ** 2).mean(axis=1).sum()
            return critic_loss, current_q_values.mean()

        (qf_loss_value, qf_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf_state.params)
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, qf_values),
            key,
        )

    @jax.jit
    def update_actor(
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        ent_coef_value: jnp.ndarray,
        observations: np.ndarray,
        key: jax.random.KeyArray,
    ):
        key, subkey = jax.random.split(key, 2)

        def actor_loss(params):
            mean, log_std = actor.apply(params, observations)
            actions, log_prob = sample_action_and_log_prob(mean, log_std, subkey)
            qf_pi = qf.apply(qf_state.params, observations, actions)
            # Take min among all critics
            min_qf_pi = jnp.min(qf_pi, axis=0)
            actor_loss = (ent_coef_value * log_prob - min_qf_pi).mean()
            return actor_loss, -log_prob.mean()

        (actor_loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, qf_state, actor_loss_value, key, entropy

    @jax.jit
    def update_temperature(ent_coef_state: TrainState, entropy: float):
        def temperature_loss(params):
            ent_coef_value = ent_coef.apply({"params": params})
            ent_coef_loss = ent_coef_value * (entropy - target_entropy).mean()
            return ent_coef_loss

        ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(ent_coef_state.params)
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss

    # add=jax.jit(buffer.add_fn,donate_argnums=(0,))
    @partial(jax.jit,donate_argnums=(2,))
    def rollout(obs, episode_stats,buffer_state, actions,handle):
        (
                episode_stats,
                handle,
                (next_obs, rewards, costs, dones, truncated, infos),
            ) = step_env_wrappeed(episode_stats, handle, actions)
            # TRY NOT TO MODIFY: execute the game and log data.
            # next_obs, rewards, dones, infos = envs.step(actions)

        buffer_state=buffer.add_fn(buffer_state, (obs[0].astype(jnp.float32), next_obs[0].astype(jnp.float32),actions[0].astype(jnp.float32),rewards.astype(jnp.float32),costs.astype(jnp.float32),dones*(1-truncated).astype(bool)))

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        return buffer_state,episode_stats,next_obs,handle



    start_time = time.time()

    reward_deque = collections.deque(maxlen=100)

    # TRY NOT TO MODIFY: start the game
    handle, (obs, info) = envs.reset()
    obs = obs.astype(jnp.float32)
    # Display progress bar if available
    generator = tqdm(range(args.total_timesteps)) if tqdm is not None else range(args.total_timesteps)
    for global_step in generator:
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.action_space.sample()])
        else:
            actions, key = sample_action(actor, actor_state, obs, key)
            actions = np.array(actions)
            # Clip due to numerical instability
            actions = np.clip(actions, -1, 1)
            # Rescale to proper domain when using squashing
            actions = unscale_action(envs.action_space, actions)

        # Part 2
        buffer_state,episode_stats,obs,handle = rollout(obs,episode_stats, buffer_state, actions,handle)
        if episode_stats.episode_lengths[0]==0:
            reward_deque.append(episode_stats.returned_episode_returns[0])
            (
                episode_stats,
                handle,
                (obs, _, _, _, _, _),
            ) = step_env_wrappeed(episode_stats, handle, actions)



        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # with jax.disable_jit():
            # actor_state, qf_state,ent_coef_state, ent_coef_value, qf_loss_value, qf_values, actor_loss_value, ent_coef_loss,key = train_step(buffer_state, qf_state, actor_state, ent_coef_state,key)
            _,(actor_state, qf_state,ent_coef_state, ent_coef_value, qf_loss_value, qf_values, actor_loss_value, ent_coef_loss,key) = train_step(buffer_state, qf_state, actor_state, ent_coef_state,key)

            # update the target networks

            if global_step % 100 == 0:
                print("losses/qf_values", qf_values.mean().item(), global_step)
                print("losses/qf_loss", qf_loss_value.item(), global_step)
                print("losses/actor_loss", actor_loss_value.item(), global_step)
                print("losses/alpha", ent_coef_value.item(), global_step)
                if tqdm is None:
                    print("SPS:", int(global_step / (time.time() - start_time)))
                print("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                print("charts/mean_return", np.average(reward_deque), global_step)
                if args.autotune:
                    print("losses/alpha_loss", ent_coef_loss.item(), global_step)
            if args.track:
                wandb.log({"global_step": global_step,"losses/qf_values": qf_values.mean().item(), "losses/qf_loss": qf_loss_value.item(), "losses/actor_loss": actor_loss_value.item(), "losses/alpha": ent_coef_value.item(),"charts/mean_return": np.average(reward_deque)})


    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state={"qf_state":qf_state,"actor_state":actor_state}
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(f"sacpolicy/{time.time()}", state, save_args=save_args)
    envs.close()


