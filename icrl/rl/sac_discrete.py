# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
# Implementation adapted from https://github.com/araffin/sbx
import argparse
import os
import random
import time
from dataclasses import dataclass
from distutils.util import strtobool
from flax.linen.initializers import constant, orthogonal
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
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import icrl.env.envs.WallGridworld# noqa: F401
import dejax
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
    parser.add_argument("--env-id", type=str, default="WallGridworld-v0",
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
    parser.add_argument("--learning-starts", type=int, default=5e2,
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
    action_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim,kernel_init=orthogonal(0.1), bias_init=constant(-0.5))(x)
        return x


class VectorCritic(nn.Module):
    n_units: int = 256
    n_critics: int = 2
    action_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
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
            action_dim=self.action_dim
        )(obs)
        return q_values


class Actor(nn.Module):
    action_dim: Sequence[int]
    n_units: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits


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
    logits = actor.apply(actor_state.params, observations)
    action = jax.random.categorical(subkey, logits)
    return action, key


@jax.jit
def sample_action_and_log_prob(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    subkey: jax.random.KeyArray,
):
    action_std = jnp.exp(log_std)
    gaussian_action = mean + action_std * jax.random.normal(subkey, shape=mean.shape)
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
    observations,next_observations,actions,rewards,dones=buffer.sample_fn(buffer_state, sample_key, args.batch_size)
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
    return actor_state,qf_state,ent_coef_state,ent_coef_value,qf_loss_value,qf_values,actor_loss_value,ent_coef_loss,entropy,key

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

    # env setup
    envs = DummyVecEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.action_space,  gym.spaces.Discrete), "only discrete action space is supported"

    if args.eval_freq > 0:
        eval_envs = make_vec_env(args.env_id, n_envs=args.n_eval_envs, seed=args.seed)

    # Create networks
    key, actor_key, qf_key, ent_key = jax.random.split(key, 4)

    obs = jnp.array([envs.observation_space.sample()])
    action = jnp.array([envs.action_space.sample()])

    buffer = dejax.uniform_replay(max_size=args.buffer_size)
    buffer_state = buffer.init_fn((obs[0],obs[0],action[0],jnp.array([0.]*envs.num_envs),jnp.array([True]*envs.num_envs)))

    actor = Actor(action_dim=envs.action_space.n)

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.policy_lr),
    )

    qf = VectorCritic(n_critics=args.n_critics,action_dim=envs.action_space.n)

    qf_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init({"params": qf_key}, obs),
        target_params=qf.init({"params": qf_key}, obs),
        tx=optax.adam(learning_rate=args.q_lr),
    )

    # Automatic entropy tuning
    if args.autotune:
        ent_coef = EntropyCoef(ent_coef_init=1.0)
        target_entropy = 0.1
        ent_coef_state = TrainState.create(
            apply_fn=ent_coef.apply,
            params=ent_coef.init(ent_key)["params"],
            tx=optax.adam(learning_rate=0.1),
        )

    else:
        ent_coef_value = jnp.array(args.alpha)

    # Define update functions here to limit the need for static argname
    # @jax.jit
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
        next_actions_logits = actor.apply(actor_state.params, next_observations)

        qf_next_values = qf.apply(qf_state.target_params, next_observations)
        next_q_values = jnp.min(qf_next_values, axis=0).clip(max=1)
        # td error + entropy term
        next_q_values = (jax.nn.softmax(next_actions_logits) * (next_q_values - ent_coef_value * jax.nn.log_softmax(next_actions_logits))).sum(-1)
        # shape is (batch_size, 1)
        target_q_values = rewards.reshape(-1) + (1 - dones.reshape(-1)) * args.gamma * next_q_values

        def mse_loss(params):
            # shape is (n_critics, batch_size, 1)
            current_q_values = qf.apply(params, observations)[:,jnp.arange(actions.shape[0]), actions]
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

    # @jax.jit
    def update_actor(
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        ent_coef_value: jnp.ndarray,
        observations: np.ndarray,
        key: jax.random.KeyArray,
    ):
        key, subkey = jax.random.split(key, 2)

        def actor_loss(params):
            logits = actor.apply(params, observations)
            qf_pi = qf.apply(qf_state.params, observations)
            # Take min among all critics
            min_qf_pi = jnp.min(qf_pi, axis=0).clip(max=1)
            actor_loss = (jax.nn.softmax(logits)*(ent_coef_value * jax.nn.log_softmax(logits) - min_qf_pi)).mean()
            return actor_loss, -(jax.nn.softmax(logits)*jax.nn.log_softmax(logits)).sum(-1).mean()

        (actor_loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, qf_state, actor_loss_value, key, entropy

    # @jax.jit
    def update_temperature(ent_coef_state: TrainState, entropy: float):
        def temperature_loss(params):
            ent_coef_value = ent_coef.apply({"params": params})
            ent_coef_loss = ent_coef_value * (entropy - target_entropy).mean()
            return ent_coef_loss

        ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(ent_coef_state.params)
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss

    envs.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device="cpu",  # force cpu device to easy torch -> numpy conversion
        handle_timeout_termination=True,
    )
    start_time = time.time()

    reward_deque = collections.deque(maxlen=100)

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    add=jax.jit(buffer.add_fn,donate_argnums=(0,))
    # Display progress bar if available
    generator = tqdm(range(args.total_timesteps)) if tqdm is not None else range(args.total_timesteps)
    for global_step in generator:
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, key = sample_action(actor, actor_state, obs, key)
            actions = np.array(actions)
            # Rescale to proper domain when using squashing
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                reward_deque.append(info["episode"]["r"])
                print(f"global_step={global_step + 1}, episodic_return={info['episode']['r']}")
                print("charts/mean_return", np.average(reward_deque), global_step)
                print("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]

        # Store the scaled action
        
        buffer_state=add(buffer_state, (obs[0], real_next_obs[0],actions[0],jnp.array(rewards),jnp.array(dones)*(1-np.array([info.get("TimeLimit.truncated", False) for info in infos]))))

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs


        if args.eval_freq > 0 and (global_step + 1) % args.eval_freq == 0:
            eval_key, agent_key = jax.random.split(eval_key, 2)
            agent = SB3Adapter(actor, actor_state, agent_key, eval_envs.action_space)
            mean_return, std_return = evaluate_policy(
                agent, eval_envs, n_eval_episodes=args.n_eval_episodes, deterministic=True
            )
            print(f"global_step={global_step + 1}, mean_eval_return={mean_return:.2f} +/- {std_return:.2f}")
            print("charts/eval_mean_ep_return", mean_return, global_step)
            print("charts/eval_std_ep_return", std_return, global_step)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # with jax.disable_jit():
            # actor_state, qf_state,ent_coef_state, ent_coef_value, qf_loss_value, qf_values, actor_loss_value, ent_coef_loss,key = train_step(buffer_state, qf_state, actor_state, ent_coef_state,key)
            _,(actor_state, qf_state,ent_coef_state, ent_coef_value, qf_loss_value, qf_values, actor_loss_value, ent_coef_loss,entropy,key) = train_step(buffer_state, qf_state, actor_state, ent_coef_state,key)

            # update the target networks

            if global_step % 100 == 0:
                print("losses/qf_values", qf_values.mean().item(), global_step)
                print("losses/qf_loss", qf_loss_value.item(), global_step)
                print("losses/entropy", entropy.mean().item(), global_step)
                
                print("losses/actor_loss", actor_loss_value.item(), global_step)
                print("losses/alpha", ent_coef_value.item(), global_step)
                if tqdm is None:
                    print("SPS:", int(global_step / (time.time() - start_time)))
                print("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    print("losses/alpha_loss", ent_coef_loss.item(), global_step)
            if args.track:
                wandb.log({"global_step": global_step,"losses/qf_values": qf_values.mean().item(), "losses/qf_loss": qf_loss_value.item(), "losses/actor_loss": actor_loss_value.item(), "losses/alpha": ent_coef_value.item(),"charts/mean_return": np.average(reward_deque)})


    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state={"qf_state":qf_state,"actor_state":actor_state}
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(f"sacpolicy/{time.time()}", state, save_args=save_args)
    envs.close()


