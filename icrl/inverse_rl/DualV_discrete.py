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

from icrl.env.cost_env import make_cost_env_no_xla
# Add progress bar if available
try:
    from tqdm.rich import tqdm
except ImportError:
    tqdm = None

@flax.struct.dataclass
class BatchData:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ICRL_discrete",
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
    parser.add_argument("--env-id", type=str, default="WallGridworld",
        help="the id of the environment to be used")
    parser.add_argument("--expert-state-path", type=str, default="sacpolicy/Reacher-expert",
        help="the path of the expert state data")
    parser.add_argument("--expert-data", type=str, default="expert_data/BiasedReacher_expert",
        help="the name of the expert data file")
    parser.add_argument("--reg-mult", type=float, default=2,
        help="the reward alpha parameter")
    parser.add_argument("--reward-alpha", type=float, default=1.0,
        help="the reward alpha parameter")
    parser.add_argument("--total-timesteps", type=int, default=50000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.8,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=1e3,
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
    parser.add_argument("--alpha", type=float, default=0.1,
        help="entropy regularization coefficient")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=False,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args




class Critic(nn.Module):
    n_units: int = 64
    action_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class ValueCritic(nn.Module):
    n_units: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class VectorCritic(nn.Module):
    n_units: int = 64
    n_critics: int = 2
    action_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray,actions: jnp.ndarray) -> jnp.ndarray:
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
        return q_values[:,jnp.arange(actions.shape[0]), actions]


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
def action_log_prob(
    action: jnp.ndarray,
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
):
    action=action.clip(-1+1e-4, 1-1e-4)
    action_std = jnp.exp(log_std)
    gaussian_action = jnp.arctanh(action)
    log_prob = (
        -0.5 * ((gaussian_action - mean) / action_std) ** 2
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - log_std
    )
    log_prob -= jnp.log((1 - action**2) + 1e-6)
    return log_prob.sum(-1)



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
def train_step(buffer_state, vf_state,qf_state, actor_state, key):
    sample_key, key = jax.random.split(key, 2)
    observations,next_observations,actions,rewards,dones=buffer.sample_fn(buffer_state, sample_key, args.batch_size)
    sample_key, key = jax.random.split(key, 2)
    Policy_Batch = BatchData(
        observations,
        actions,
        next_observations,
        rewards,
        dones,
    )
    expert_indice = jax.random.randint(
        sample_key, minval=0, maxval=len(expert_obs), shape=(args.batch_size,)
    )
    Expert_Batch = BatchData(
        expert_obs[expert_indice],
        expert_action[expert_indice],
        expert_next_obs[expert_indice],
        expert_rewards[expert_indice],
        expert_dones[expert_indice],
    )
    qf_state, (qf_loss_value, qf_values), key = update_critic(
        actor_state,
        qf_state,
        vf_state,
        Policy_Batch,
        Expert_Batch,
        key,
    )
    vf_state, (vf_loss_value, vf_values, qf_values), key = update_value_critic(
        qf_state,
        vf_state,
        Policy_Batch,
        Expert_Batch,
        key,
    )
    (actor_state, qf_state, actor_loss_value, key, entropy) = update_actor(
                actor_state,
                vf_state,
                qf_state,
                ent_coef_value,
                Policy_Batch,
                Expert_Batch,
                key,
            )
    qf_state = soft_update(args.tau, qf_state)

    return actor_state,vf_state,qf_state,qf_loss_value,qf_values,vf_loss_value, vf_values,actor_loss_value,key

def get_data():
    expert_action = jnp.array([0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 3])
    expert_obs = jnp.array([[0, 0],
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 3],
    [2, 3],
    [3, 3],
    [4, 3],
    [4, 2],
    [4, 1],
    [4, 0]])
    expert_next_obs = jnp.array([[ 0,  1],
    [ 0,  2],
    [ 0,  3],
    [ 1,  3],
    [ 2,  3],
    [ 3,  3],
    [ 4,  3],
    [ 4,  2],
    [ 4,  1],
    [ 4,  0],
    [-1, -1]])
    expert_dones = jnp.array([False, False, False, False, False, False, False, False, False, False,
    True])
    return expert_obs.astype(jnp.float32),expert_action,expert_next_obs.astype(jnp.float32),expert_dones,expert_dones.astype(jnp.float32)

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

    reg_mult=args.reg_mult
    Q_max=1.0 / (reg_mult * (1 - args.gamma))
    Q_min=-1.0 / (reg_mult * (1 - args.gamma))
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
    action = jnp.array([envs.action_space.sample()])

    buffer = dejax.uniform_replay(max_size=args.buffer_size)
    buffer_state = buffer.init_fn((obs[0],obs[0],action[0],jnp.array(0.,dtype=jnp.float32),jnp.array(True)))

    actor = Actor(action_dim=envs.action_space.n)

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.policy_lr),
    )
    qf = VectorCritic(n_critics=args.n_critics,action_dim=envs.action_space.n)

    qf_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init({"params": qf_key}, obs,action),
        target_params=qf.init({"params": qf_key}, obs,action),
        tx=optax.adam(learning_rate=args.q_lr),
    )

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state = {"qf_state": qf_state, "actor_state": actor_state}
    # if args.reward_alpha!=0.0:
    # state = checkpointer.restore(args.expert_state_path, item=state)
    # actor_state = state["actor_state"]
    expert_data = checkpointer.restore(args.expert_data)
    expert_obs,expert_action,expert_next_obs,expert_dones,expert_rewards=get_data()
    expert_obs=jnp.array(envs.envs[0].state_to_onehot(expert_obs.astype(int)))

    expert_indice = jax.random.randint(
        key, minval=0, maxval=len(expert_obs), shape=(int(len(expert_obs)/2)+1,)
    )
    expert_obs=expert_obs[expert_indice]
    expert_action=expert_action[expert_indice]
    expert_next_obs=expert_next_obs[expert_indice]
    expert_rewards=expert_rewards[expert_indice]
    expert_dones=expert_dones[expert_indice]


    expert_next_obs=jnp.array(envs.envs[0].state_to_onehot(expert_next_obs.astype(int)))
    vf=ValueCritic()
    vf_state = TrainState.create(
        apply_fn=vf.apply,
        params=vf.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.policy_lr),
    )


    qf = VectorCritic(n_critics=1,action_dim=envs.action_space.n)

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

    @jax.jit
    def update_critic(
        actor_state: RLTrainState,        
        qf_state: RLTrainState,
        vf_state: RLTrainState,
        Policy_Batch: BatchData,
        Expert_Batch: BatchData,
        key: jax.random.KeyArray,
    ):
        alpha=args.reward_alpha
        key, subkey = jax.random.split(key, 2)
        key1, key2 = jax.random.split(subkey, 2)
        policy_next_values =vf.apply(vf_state.params,Policy_Batch.next_observations)
        key1, key2 = jax.random.split(key1, 2)
        expert_next_values =vf.apply(vf_state.params,Expert_Batch.next_observations)
        key1, key2 = jax.random.split(key1, 2)

        def mse_loss(params):
            expert_current_Q = qf.apply(
                params, Expert_Batch.observations, Expert_Batch.actions
            )
            policy_current_Q = qf.apply(
                params, Policy_Batch.observations, Policy_Batch.actions
            )
            policy_discounted_rewards=alpha*Policy_Batch.rewards+(-1/reg_mult)+((1 - Policy_Batch.dones).reshape(-1, 1)*args.gamma*policy_next_values).squeeze(-1)
            expert_discounted_rewards=alpha*Expert_Batch.rewards+(1/reg_mult)+((1 - Expert_Batch.dones).reshape(-1, 1)*args.gamma*expert_next_values).squeeze(-1)
            loss = ((expert_current_Q - expert_discounted_rewards)**2).mean()+((policy_current_Q - policy_discounted_rewards)**2).mean()
            return loss, jnp.concatenate([expert_current_Q, policy_current_Q]).mean()

        (qf_loss_value, qf_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            qf_state.params
        )
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, qf_values),
            key,
        )

    @jax.jit
    def update_value_critic(
        qf_state: RLTrainState,
        vf_state: RLTrainState,
        Policy_Batch: BatchData,
        Expert_Batch: BatchData,
        key: jax.random.KeyArray,
    ):
        Batch=jax.tree_map(lambda x,y:jnp.concatenate([x,y]),Policy_Batch,Expert_Batch)
        current_Q = qf.apply(
            qf_state.target_params, Batch.observations, Batch.actions
        )
        lamb=0.7
        def mse_loss(params):
            current_V = vf.apply(
                params, Batch.observations
            )
            diff=current_Q.squeeze(0)-current_V.squeeze(1)
            loss=(-(1-lamb)*diff+lamb*(diff+diff**2/4).clip(0)).mean()
            return loss, (current_V.mean(),current_Q.mean())

        (vf_loss_value, (vf_values, qf_values)), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            vf_state.params
        )
        vf_state = vf_state.apply_gradients(grads=grads)

        return (
            vf_state,
            (vf_loss_value, vf_values, qf_values),
            key,
        )




    @jax.jit
    def update_actor(
        actor_state: RLTrainState,
        vf_state: RLTrainState,
        qf_state: RLTrainState,
        ent_coef_value: jnp.ndarray,
        Policy_Batch:BatchData,
        Expert_Batch:BatchData,
        key: jax.random.KeyArray,
    ):
        key, subkey = jax.random.split(key, 2)
        Batch=Policy_Batch

        def actor_loss(params):
            logits = actor.apply(params, Batch.observations)
            prob = jax.nn.softmax(logits)
            log_prob = jnp.log(prob)
            action_log_prob = log_prob[jnp.arange(Batch.actions.shape[0]), Batch.actions]
            vf_pi = vf.apply(vf_state.params, Batch.observations)
            qf_pi = qf.apply(qf_state.params, Batch.observations, Batch.actions)
            # Take min among all critics
            min_qf_pi = jnp.min(qf_pi, axis=0)
            actor_loss = -(jnp.exp(((min_qf_pi-vf_pi.flatten()))*10).clip(max=100.).flatten() * action_log_prob ).mean()
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
    @partial(jax.jit, donate_argnums=(0))
    def rollout(buffer_state, obs, next_obs, actions, rewards, dones):
        rewards = jnp.nan_to_num(rewards.astype(jnp.float32), neginf=0, posinf=0)
        buffer_state = buffer.add_fn(
            buffer_state, (obs, next_obs, actions, rewards, dones)
        )
        return buffer_state



    start_time = time.time()

    reward_deque = collections.deque(maxlen=10)
    cost_deque = collections.deque(maxlen=10)
    length_deque = collections.deque(maxlen=10)

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset()
    obs = obs.astype(np.float32)
    # Display progress bar if available
    generator = tqdm(range(int(args.learning_starts)))
    cum_reward = 0
    cum_cost = 0
    lengths = 0
    for global_step in generator:
        actions = jnp.array([envs.action_space.sample()])
        (next_obs, rewards, dones, truncated, infos) = envs.step(actions.__array__())
        costs = cost_function(next_obs, rewards, dones, truncated, infos)
        next_obs = next_obs.astype(np.float32)

        buffer_state= rollout(
            buffer_state,
            obs[0],
            next_obs[0],
            actions[0],
            rewards[0],
            (dones * (1 - truncated).astype(bool))[0],
        )
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
            (obs, rewards, dones, truncated, infos) = envs.step(actions.__array__())
            obs = obs.astype(np.float32)

    def train_step_body(carry,step):
        buffer_state, actor_state,vf_state,qf_state, key = carry
        _, (actor_state,vf_state,qf_state,qf_loss_value,qf_values,vf_loss_value, vf_values,actor_loss_value,key) = train_step(buffer_state, vf_state,qf_state, actor_state, key)
        return (buffer_state, actor_state,vf_state,qf_state,key), (qf_loss_value,qf_values.mean(),vf_loss_value, vf_values.mean(),actor_loss_value)



    for global_step in range(int((args.total_timesteps-args.learning_starts)/50)):
        for i in range(50):
            actions, key = sample_action(actor, actor_state, obs, key)
            (next_obs, rewards, dones, truncated, infos) = envs.step(actions.__array__())
            costs = cost_function(next_obs, rewards, dones, truncated, infos)
            next_obs = next_obs.astype(np.float32)

            buffer_state= rollout(
                buffer_state,
                obs[0],
                next_obs[0],
                actions[0],
                rewards[0],
                (dones * (1 - truncated).astype(bool))[0],
            )
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
                (obs, rewards, dones, truncated, infos) = envs.step(actions.__array__())
                obs = obs.astype(np.float32)

        (buffer_state, actor_state,vf_state,qf_state, key), (qf_loss_value,qf_values,vf_loss_value, vf_values,actor_loss_value) = jax.lax.scan(
            train_step_body,
            (buffer_state, actor_state,vf_state,qf_state, key),
            (),
            length=50,
        )
        log_data = {
            "losses/vf_values": vf_values.mean().item(),
            "losses/vf_loss_value": vf_loss_value.mean().item(),
            "losses/qf_values": qf_values.mean().item(),
            "losses/qf_loss": qf_loss_value.mean().item(),
            "losses/actor_loss": actor_loss_value.mean().item(),
            "losses/alpha": ent_coef_value.mean().item(),
            "charts/SPS": int(global_step / (time.time() - start_time)),
            "charts/mean_return": np.average(reward_deque),
            "charts/mean_cost": np.average(cost_deque),
            "charts/mean_length": np.average(length_deque),
        }

        # Print the data
        for k, v in log_data.items():
            print(k, v, global_step)

        # Log the data if args.track is True
        if args.track:
            # Assuming you have imported the required libraries (e.g., wandb)
            wandb.log({"global_step": global_step, **log_data})

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state={"qf_state":qf_state,"vf_state":vf_state,"actor_state":actor_state}
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(f"sacpolicy/{args.env_id}/{run_name}", state, save_args=save_args)
    # envs.close()


