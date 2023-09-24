import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product
import pickle
import argparse
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax.numpy as jnp
import jax
import optax
from jax import random

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="random seed",default=0)
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)


def generate_random_path(grid, start, end):
    def find_path(current_position, visited):
        if current_position == end:
            return [current_position]

        possible_moves = []
        row, col = current_position
        rows, cols = grid.shape

        if row > 0 and grid[row - 1, col] == 1 and (row - 1, col) not in visited:
            possible_moves.append((-1, 0))
        if row < rows - 1 and grid[row + 1, col] == 1 and (row + 1, col) not in visited:
            possible_moves.append((1, 0))
        if col > 0 and grid[row, col - 1] == 1 and (row, col - 1) not in visited:
            possible_moves.append((0, -1))
        if col < cols - 1 and grid[row, col + 1] == 1 and (row, col + 1) not in visited:
            possible_moves.append((0, 1))

        if (
            row > 0
            and col > 0
            and grid[row - 1, col - 1] == 1
            and (row - 1, col - 1) not in visited
        ):
            possible_moves.append((-1, -1))
        if (
            row > 0
            and col < cols - 1
            and grid[row - 1, col + 1] == 1
            and (row - 1, col + 1) not in visited
        ):
            possible_moves.append((-1, 1))
        if (
            row < rows - 1
            and col > 0
            and grid[row + 1, col - 1] == 1
            and (row + 1, col - 1) not in visited
        ):
            possible_moves.append((1, -1))
        if (
            row < rows - 1
            and col < cols - 1
            and grid[row + 1, col + 1] == 1
            and (row + 1, col + 1) not in visited
        ):
            possible_moves.append((1, 1))

        if not possible_moves:
            return []

        next_move = possible_moves[np.random.choice(len(possible_moves))]
        new_position = (
            current_position[0] + next_move[0],
            current_position[1] + next_move[1],
        )
        path = find_path(new_position, visited + [current_position])

        return [current_position] + path if path is not None else None

    path = find_path(start, [])
    return np.array(path) if path is not None else None


def saveimg(data, save_path):
    fig, ax = plt.subplots()

    im = ax.imshow(data, cmap="gray", alpha=0.9)

    # Add a title
    ax.set_title("Estimate cost V")

    # Add labels to the x-axis and y-axis
    ax.set_xlabel("X-Axis")
    ax.set_ylabel("Y-Axis")

    # Add a colorbar
    fig.colorbar(im)
    for (j, i), label in np.ndenumerate(data):
        ax.text(i, j, "{:.2f}".format(label), ha="center", va="center")
    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    plt.savefig(save_path)
    plt.close()

gird_shape = (7, 7)
settingname = "setting2"

expert_trajs = np.load(f"expert_data/grid_world/{settingname}/expert_trajs.npz")
expert_trajs = [expert_trajs[name] for name in expert_trajs.files]

start_point = tuple(expert_trajs[0][0].tolist())
end_point = tuple(expert_trajs[0][-1].tolist())
d_ab = np.zeros(gird_shape)
d_ab[end_point] = 1

def generate_traj():
    sample_trajs = []
    random_path = [
        generate_random_path(np.ones(gird_shape), start_point, end_point)
        for i in range(20)
    ]
    sample_trajs += random_path
    if (0, 6) != end_point:
        random_path = [
            generate_random_path(np.ones(gird_shape), (0, 6), end_point)
            for i in range(10)
        ]
        sample_trajs += random_path
    if (6, 0) != end_point:
        random_path = [
            generate_random_path(np.ones(gird_shape), (6, 0), end_point)
            for i in range(10)
        ]
        sample_trajs += random_path
    if (0, 0) != end_point:
        random_path = [
            generate_random_path(np.ones(gird_shape), (0, 0), end_point)
            for i in range(10)
        ]
        sample_trajs += random_path
    if (6, 6) != end_point:
        random_path = [
            generate_random_path(np.ones(gird_shape), (6, 6), end_point)
            for i in range(10)
        ]
        sample_trajs += random_path
    return sample_trajs
def calculate_traj_distances(traj):
    r = (traj[1:] == end_point).all(axis=1) - 1

    d = np.power(gamma, np.arange(len(traj) - 1))
    return d, r

gamma = 1

beta = 0.5




rho_E = np.zeros((*gird_shape, *gird_shape))

for traj in expert_trajs:
    rho_E[traj[:-1, 0], traj[:-1:, 1], traj[1:, 0], traj[1:, 1]] += 1

state_ocuupy_E = np.zeros(gird_shape)
for traj in expert_trajs:
    state_ocuupy_E[traj[:-1, 0], traj[:-1, 1]] += 1


expert_d = rho_E / rho_E.sum((-1, -2)).clip(1)[..., None, None]
expert_d[(expert_d < 1) & (expert_d > 0)] = 0.5


target_E = jnp.concatenate(
    [
        expert_d[traj[:-1, 0], traj[:-1:, 1], traj[1:, 0], traj[1:, 1]]
        for traj in expert_trajs
    ]
)


def get_all_data(trajs):
    d, r = list(zip(*[calculate_traj_distances(traj) for traj in trajs]))
    return np.concatenate(d), np.concatenate(r)

def calculate_y(obs, obs_next, V):
    y = gamma * V[obs_next[:, 0], obs_next[:, 1]] - V[obs[:, 0], obs[:, 1]]
    return y, V[obs_next[:, 0], obs_next[:, 1]], V[obs[:, 0], obs[:, 1]]




def f_div(x):
    return (x - 1) ** 2


def omega_star(y):
    return (y / 2 + 1).clip(0)


def fp_star(y):
    return omega_star(y) * y - f_div(omega_star(y))
sample_trajs = generate_traj()

target_S = jnp.concatenate(
    [
        expert_d[traj[:-1, 0], traj[:-1:, 1], traj[1:, 0], traj[1:, 1]]
        for traj in sample_trajs
    ]
)

show_S = np.concatenate(
    [state_ocuupy_E[traj[:-1:, 0], traj[:-1, 1]] != 0 for traj in sample_trajs]
)
d_E, r_E = get_all_data(expert_trajs)
d_S, r_S = get_all_data(sample_trajs)

obs_E = np.concatenate([traj[:-1] for traj in expert_trajs])
obs_next_E = np.concatenate([traj[1:] for traj in expert_trajs])

obs_S = np.concatenate([traj[:-1] for traj in sample_trajs])
obs_next_S = np.concatenate([traj[1:] for traj in sample_trajs])


def get_V(V):
    y_E, V_E, next_V_E = calculate_y(obs_E, obs_next_E, V)
    y_S, V_S, next_V_S = calculate_y(obs_S, obs_next_S, V)
    return (y_E, V_E, next_V_E), (y_S, V_S, next_V_S)


def train(p_lower=0.001, lamb=0.45, max_constraint=800):
    @jax.jit
    def dualrewardV(reward_V, V):

        (y_E, V_E, next_V_E), (y_S, V_S, next_V_S) = get_V(reward_V)
        (cost_y_E, _, _), (cost_y_S, _, _) = get_V(V)
        cost_y_E = cost_y_E.clip(-1, 0)
        cost_y_S = cost_y_S.clip(-1, 0)
        return (
            (1 - lamb) * (beta * V_E.mean() + (1 - beta) * V_S.mean())
            + lamb
            * (
                beta * (fp_star(y_E + r_E + cost_y_E) * d_E).mean()
                + (1 - beta) * (fp_star(y_S + r_S + cost_y_S) * d_S).mean()
            )
            + 10 * (jnp.abs((reward_V * d_ab))).sum()
        )

    @jax.jit
    def dualV(V, reward_V, V_alpha=1):

        (y_E, V_E, next_V_E), (y_S, V_S, next_V_S) = get_V(reward_V)
        # V=V.clip(max=0)
        (cost_y_E, cost_V_E, next_cost_V_E), (
            cost_y_S,
            cost_V_S,
            next_cost_V_S,
        ) = get_V(V)
        pi_E = omega_star(y_E + r_E + cost_y_E.clip(-1, max=0))
        pi_S = omega_star(y_S + r_S + cost_y_S.clip(-1, max=0))
        cost = 1 * (
            (1 - beta)
            * (
                d_S
                * (
                    jnp.abs(cost_y_S)
                    + jnp.exp(cost_y_S.clip(0) - cost_y_S.clip(max=-1))
                )
            ).mean()
            + (beta)
            * (
                d_E
                * (
                    jnp.abs(cost_y_E)
                    + jnp.exp(50 * cost_y_E.clip(0) - cost_y_E.clip(max=-1))
                )
            ).mean()
        ) + V_alpha * 100 * (
            (1 - beta) * (d_S * (jnp.abs(cost_V_S) + next_cost_V_S.clip(0))).mean()
            + (beta) * (d_E * (jnp.abs(cost_V_E) + next_cost_V_E.clip(0))).mean()
        )
        return (
            beta * -(pi_E * jnp.log(target_E.clip(p_lower))).mean()
            + (1 - beta) * (show_S * -(pi_S * jnp.log(target_S.clip(p_lower)))).mean()
            + cost * 0.00001
            + 10 * (jnp.abs((V * d_ab))).sum()
        )

    @jax.jit
    def fit():
        optimizer = optax.adam(learning_rate=1e-2)
        V = jnp.zeros(gird_shape)
        reward_V = jnp.zeros(gird_shape)
        opt_state = optimizer.init(V)
        reward_opt_state = optimizer.init(reward_V)

        def reward_step(reward_V, opt_state, V):
            loss_value, grads = jax.value_and_grad(dualrewardV)(reward_V, V)
            updates, opt_state = optimizer.update(grads, opt_state, reward_V)
            reward_V = optax.apply_updates(reward_V, updates)
            return reward_V, opt_state

        def step(V, reward_V, opt_state, V_alpha):
            loss_value, grads = jax.value_and_grad(dualV)(
                V, reward_V, V_alpha.clip(max=max_constraint)
            )
            updates, opt_state = optimizer.update(grads, opt_state, V)
            V = optax.apply_updates(V, updates)
            return V, opt_state

        reward_V, opt_state = jax.lax.fori_loop(
            0,
            10000,
            lambda i, val: reward_step(val[0], val[1], V),
            (reward_V, opt_state),
        )

        def cost_learning(i, carry):
            V, reward_V, opt_state, reward_opt_state = carry
            V, opt_state = step(V, reward_V, opt_state, i / 100)
            reward_V, reward_opt_state = reward_step(reward_V, opt_state, V)
            return V, reward_V, opt_state, reward_opt_state

        V, reward_V, opt_state, reward_opt_state = jax.lax.fori_loop(
            0, 100000, cost_learning, (V, reward_V, opt_state, reward_opt_state)
        )

        return V, reward_V

    return fit()


# print(res)
key = random.PRNGKey(0)  # Random seed is explicit in JAX

# p_lowers=[0.001]
# lambs = [0.1]
# max_constraints=[100]


p_lowers=[0.001,0.005,0.01,0.05]
lambs = [0.1,0.3,0.5,0.7,0.9]
max_constraints=[100,400,700,1000]

products = list(product(p_lowers, lambs, max_constraints))

data = {}
for p_lower, lamb, max_constraint in products:
    V, reward_V = train(p_lower, lamb, max_constraint)
    data[f"p_lower_{p_lower}_lamb_{lamb}_max_constraint_{max_constraint}"] = (
        V.tolist(),
        reward_V.tolist(),
    )
    # saveimg(
    #     reward_V,
    #     f"grid_img/{settingname}/p_lower_{p_lower}_lamb_{lamb}_max_constraint_{max_constraint}/reward_V_{seed}.jpg",
    # )
    # saveimg(
    #     V,
    #     f"grid_img/{settingname}/p_lower_{p_lower}_lamb_{lamb}_max_constraint_{max_constraint}/V_{seed}.jpg",
    # )

with open(f"grid_img/{settingname}/data_{seed}", "wb") as f:
    pickle.dump(data, f)
