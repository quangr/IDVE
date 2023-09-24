import jax.numpy as jnp
def split_data(expert_data):
    expert_action = jnp.array(expert_data["action"]).astype(jnp.float32)
    expert_obs = jnp.array(expert_data["traj_obs"]).astype(jnp.float32)
    expert_next_obs = jnp.array(expert_data["traj_next_obs"]).astype(jnp.float32)
    expert_dones = jnp.array(expert_data["dones"]).astype(jnp.float32)
    expert_rewards = jnp.array(expert_data["rewards"]).astype(jnp.float32)
    return expert_obs, expert_next_obs, expert_action, expert_rewards, expert_dones
