import numpy as np
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax.numpy as jnp
import jax
import optax
from jax import random
import matplotlib.pyplot as plt

seed = 0
np.random.seed(seed)


def create_coordinate_set(x1, y1, x2, y2):
    if x1 == x2:
        coords = [
            (x1, i)
            for i in range(y1, y2 + (-1 if y1 > y2 else 1), -1 if y1 > y2 else 1)
        ]
    elif y1 == y2:
        coords = [
            (i, y1)
            for i in range(x1, x2 + (-1 if x1 > x2 else 1), -1 if x1 > x2 else 1)
        ]
    return np.array(coords)


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
expert_trajs = np.load("expert_data/grid_world/setting1/expert_trajs.npz")
expert_trajs = [expert_trajs[name] for name in expert_trajs.files]
# sample_trajs = np.load("expert_data/grid_world/setting2/sample_trajs.npz")

# sample_trajs = [sample_trajs[name] for name in sample_trajs.files]


start_point = tuple(expert_trajs[0][0].tolist())
end_point = tuple(expert_trajs[0][-1].tolist())

sample_trajs = []
random_path = [
    generate_random_path(np.ones(gird_shape), start_point, end_point) for i in range(50)
]
sample_trajs += random_path
if (0, 6) != end_point:
    random_path = [
        generate_random_path(np.ones(gird_shape), (0, 6), end_point) for i in range(10)
    ]
    sample_trajs += random_path
if (6, 0) != end_point:
    random_path = [
        generate_random_path(np.ones(gird_shape), (6, 0), end_point) for i in range(10)
    ]
    sample_trajs += random_path
if (0, 0) != end_point:
    random_path = [
        generate_random_path(np.ones(gird_shape), (0, 0), end_point) for i in range(10)
    ]
    sample_trajs += random_path
if (6, 6) != end_point:
    random_path = [
        generate_random_path(np.ones(gird_shape), (6, 6), end_point) for i in range(10)
    ]
    sample_trajs += random_path
d_0 = np.zeros(gird_shape)
d_0[start_point] = 1

d_ab = np.zeros(gird_shape)
d_ab[end_point] = 1


gamma = 0.99

beta = 0.5


def calculate_traj_distances(traj):
    r = (traj[1:] == end_point).all(axis=1) - 1
    d = np.power(gamma, np.arange(len(traj) - 1))
    return d, r


rho_S = np.zeros((*gird_shape, *gird_shape))
rho_E = np.zeros((*gird_shape, *gird_shape))

for traj in expert_trajs:
    rho_E[traj[:-1, 0], traj[:-1:, 1], traj[1:, 0], traj[1:, 1]] += 1
for traj in sample_trajs:
    rho_S[traj[:-1, 0], traj[:-1:, 1], traj[1:, 0], traj[1:, 1]] += 1

state_ocuupy_E = np.zeros(gird_shape)
for traj in expert_trajs:
    state_ocuupy_E[traj[:-1, 0], traj[:-1, 1]] += 1


rho = (beta * rho_E + (1 - beta) * rho_S) / (
    (beta * rho_E + (1 - beta) * rho_S).sum((-1, -2))
).clip(0.1)[..., None, None]
expert_d = rho_E / rho_E.sum((-1, -2)).clip(1)[..., None, None]
expert_d[(expert_d < 1) & (expert_d > 0)] = 0.5
expert_d=jnp.array(expert_d)

# weight_e=[]
# for traj in expert_trajs:
#     weight_e.append((expert_d[traj[:-1, 0], traj[:-1:, 1]]!=0).sum((-1,-2)).prod())
# print(weight_e)
# target_E = np.concatenate(
#     [
#         (expert_d / rho)[traj[:-1, 0], traj[:-1:, 1], traj[1:, 0], traj[1:, 1]]
#         for traj in expert_trajs
#     ]
# )
# target_S = np.concatenate(
#     [
#         (expert_d / rho)[traj[:-1, 0], traj[:-1:, 1], traj[1:, 0], traj[1:, 1]]
#         for traj in sample_trajs
#     ]
# )
target_E = np.concatenate(
    [
        expert_d[traj[:-1, 0], traj[:-1:, 1], traj[1:, 0], traj[1:, 1]]
        for traj in expert_trajs
    ]
)
target_S = np.concatenate(
    [
        expert_d[traj[:-1, 0], traj[:-1:, 1], traj[1:, 0], traj[1:, 1]]
        for traj in sample_trajs
    ]
)


def get_all_data(trajs, rho):
    d_E, r_E = list(zip(*[calculate_traj_distances(traj) for traj in trajs]))
    d_E = np.concatenate(d_E)
    r_E = np.concatenate(r_E)
    rho_E = np.concatenate(
        [rho[traj[:-1, 0], traj[:-1:, 1], traj[1:, 0], traj[1:, 1]] for traj in trajs]
    )
    return rho_E, d_E, r_E


show_S = np.concatenate(
    [state_ocuupy_E[traj[:-1:, 0], traj[:-1, 1]] != 0 for traj in sample_trajs]
)
rho_E, d_E, r_E = get_all_data(expert_trajs, rho)
rho_S, d_S, r_S = get_all_data(sample_trajs, rho)

obs_E = np.concatenate([traj[:-1] for traj in expert_trajs])
obs_next_E = np.concatenate([traj[1:] for traj in expert_trajs])

obs_S = np.concatenate([traj[:-1] for traj in sample_trajs])
obs_next_S = np.concatenate([traj[1:] for traj in sample_trajs])


def calculate_y(obs, obs_next, V):
    y = gamma * V[obs_next[:, 0], obs_next[:, 1]] - V[obs[:, 0], obs[:, 1]]
    return y,V[obs[:, 0], obs[:, 1]], V[obs_next[:, 0], obs_next[:, 1]]

def get_V(V):
    y_E, V_E, next_V_E = calculate_y(obs_E, obs_next_E, V)
    y_S, V_S, next_V_S = calculate_y(obs_S, obs_next_S, V)
    return (y_E, V_E, next_V_E), (y_S, V_S, next_V_S)


def get_Q(Q):
    Q_E = Q[obs_E[:, 0], obs_E[:, 1], obs_next_E[:, 0], obs_next_E[:, 1]]
    Q_S = Q[obs_S[:, 0], obs_S[:, 1], obs_next_S[:, 0], obs_next_S[:, 1]]
    return Q_E, Q_S


lamb = 0.55

# def f_div(x):
#     return x*jnp.log(x)

# def omega_star(y):
#     return jnp.exp(y-1)


def f_div(x):
    return (x - 1) ** 2


def omega_star(y):
    return (y / 2 + 1).clip(0)


def fp_star(y):
    return omega_star(y) * y - f_div(omega_star(y))


p_lower = 0.005


def dualrewardV(reward_V,reward_Q, V):
    Q_E, Q_S = get_Q(reward_Q)
    # def fp_star(y):
    #     return (y + y**2 / 2).clip(0)
    (y_E, V_E, next_V_E), (y_S, V_S, next_V_S) = get_V(reward_V)
    return (
        (1 - lamb) * (beta * V_E.mean() + (1 - beta) * V_S.mean())
        + lamb
        * (
            beta * (fp_star(Q_E - V_E) * d_E).mean()
            + (1 - beta) * (fp_star(Q_S - V_S) * d_S).mean()
        )
    )


def dualV(V, reward_Q,reward_V, Q,key, V_alpha=1):
    Q_E, Q_S = get_Q(reward_Q)
    cost_Q_E, cost_Q_S = get_Q(Q)
    max_obs=((V[None,None]+reward_Q).reshape(7*7,-1)).argmax(1).reshape(7,7)
    max_obs_next_E = max_obs[obs_E[:, 0], obs_E[:, 1]]
    max_obs_next_E=jnp.stack(jnp.unravel_index(max_obs_next_E, gird_shape)).T
    max_y_E, max_V_E, max_next_V_E = calculate_y(obs_E, max_obs_next_E, V)
    max_Q_E=reward_Q[obs_E[:, 0], obs_E[:, 1], max_obs_next_E[:, 0], max_obs_next_E[:, 1]]
    max_cost_Q_E=Q[obs_E[:, 0], obs_E[:, 1], max_obs_next_E[:, 0], max_obs_next_E[:, 1]]
    (y_E, V_E, next_V_E), (y_S, V_S, next_V_S) = get_V(reward_V)
    (cost_y_E, cost_V_E, next_cost_V_E), (cost_y_S, cost_V_S, next_cost_V_S) = get_V(V)

    target_max=expert_d[obs_E[:, 0], obs_E[:, 1], max_obs_next_E[:, 0], max_obs_next_E[:, 1]]
    pi_E = omega_star(Q_E -V_E+  (cost_Q_E-cost_V_E).clip(-1, 0))
    max_pi_E = omega_star(max_Q_E -V_E+  (max_cost_Q_E-cost_V_E).clip(-1, 0))
    pi_S = omega_star(Q_S -V_S+  (cost_Q_S-cost_V_S).clip(-1, 0))
    cost = 1 * (
        (1 - beta)
        * (d_S * (jnp.abs(cost_y_S) + jnp.exp(cost_y_S.clip(0) - cost_y_S.clip(max=-1)))).mean()
        + (beta)
        * (d_E * (jnp.abs(cost_y_E) + jnp.exp(50*cost_y_E.clip(0) - cost_y_E.clip(max=-1)))).mean()
    )+V_alpha*(100* (
        (1 - beta)
        * (d_S * (jnp.abs(cost_V_S))).mean()
        + (beta)
        * (d_E * (jnp.abs(cost_V_E))).mean()
    +V.clip(0).sum())
)
    return (
        beta * -(Q_E -V_E+  (cost_Q_E-cost_V_E).clip(-1, 0)).mean()
        +(1-beta)*(show_S*(target_S==0)*(Q_S -V_S+  (cost_Q_S-cost_V_S).clip(-1, 0)).clip(-0.4)).mean()
        # -beta*((pi_E*jnp.log(target_E.clip(p_lower))).mean())#-(max_pi_E*jnp.log(target_max.clip(p_lower))).mean()
        + cost * 0.00001
        # +(1-beta)*(show_S*-(pi_S*jnp.log(target_S.clip(p_lower)))).mean()
        + 1000 * (jnp.abs((V * d_ab))).sum()
    )


def dualrewardQ(reward_Q, reward_V, V):
    Q_E, Q_S = get_Q(reward_Q)    
    (y_E, V_E, next_V_E), (y_S, V_S, next_V_S) = get_V(reward_V)
    (cost_y_E, cost_V_E, next_cost_V_E), (cost_y_S, cost_V_S, next_cost_V_S) = get_V(V)
    return beta*((Q_E - ((r_E!=0)*gamma * next_V_E + r_E + cost_y_E.clip(-1, 0))) ** 2).mean() + (1-beta)*((
        Q_S - ((r_S!=0)*gamma * next_V_S + r_S + cost_y_S.clip(-1, 0))
    ) ** 2).mean()#+ 10 * (jnp.abs((reward_Q[...,end_point[0],end_point[1]]))).sum()


def dualQ(Q, V):
    cost_Q_E, cost_Q_S = get_Q(Q)
    (cost_y_E, cost_V_E, next_cost_V_E), (cost_y_S, cost_V_S, next_cost_V_S) = get_V(V)
    return ((cost_Q_E - gamma*next_cost_V_E) ** 2).mean() + ((cost_Q_S - gamma*next_cost_V_S) ** 2).mean()


def fit(V, optimizer):
    reward_V = jnp.zeros(gird_shape)
    reward_Q = jnp.zeros((*gird_shape, *gird_shape))
    Q = jnp.zeros((*gird_shape, *gird_shape))
    opt_state = optimizer.init(V)
    reward_opt_state = optimizer.init(reward_V)
    q_opt_state = optimizer.init(Q)
    reward_q_opt_state = optimizer.init(reward_Q)

    @jax.jit
    def reward_Q_step(reward_Q, reward_V, V, opt_state):
        loss_value, grads = jax.value_and_grad(dualrewardQ)(reward_Q, reward_V, V)
        updates, opt_state = optimizer.update(grads, opt_state, reward_Q)
        reward_Q = optax.apply_updates(reward_Q, updates)
        return reward_Q, opt_state, loss_value

    @jax.jit
    def Q_step(Q, V, opt_state):
        loss_value, grads = jax.value_and_grad(dualQ)(Q, V)
        updates, opt_state = optimizer.update(grads, opt_state, Q)
        Q = optax.apply_updates(Q, updates)
        return Q, opt_state, loss_value

    @jax.jit
    def reward_step(reward_V,reward_Q, V, opt_state):
        loss_value, grads = jax.value_and_grad(dualrewardV)(reward_V,reward_Q, V)
        updates, opt_state = optimizer.update(grads, opt_state, reward_V)
        reward_V = optax.apply_updates(reward_V, updates)
        return reward_V, opt_state, loss_value

    @jax.jit
    def step(V, reward_Q, reward_V, Q,key, opt_state, V_alpha):
        (key,)=jax.random.split(key, 1)
        loss_value, grads = jax.value_and_grad(dualV)(
            V, reward_Q, reward_V, Q,key, V_alpha.clip(max=1000)
        )
        updates, opt_state = optimizer.update(grads, opt_state, V)
        V = optax.apply_updates(V, updates)
        return V, opt_state, key,loss_value

    key = random.PRNGKey(758493)  # Random seed is explicit in JAX
    is_reward = 0
    reward_V = jnp.zeros(gird_shape)
    reward_Q = -100*jnp.ones((*gird_shape, *gird_shape))
    _reward_Q=reward_Q
    Q = -100*jnp.ones((*gird_shape, *gird_shape))
    _Q=Q
    for i in range(100000):
        reward_Q, reward_q_opt_state, loss_value = reward_Q_step(
           reward_Q, reward_V, V, reward_q_opt_state
        )
        reward_V, reward_opt_state, loss_value = reward_step(
            reward_V,_reward_Q, V, reward_opt_state
        )
        _reward_Q=0.99*_reward_Q+0.01*reward_Q
    old_reward_V = reward_V
    old_V = V
    for epoch in range(50):
        V=V-V.max()
        for i in range(5000):
            V, opt_state,  key,loss_value= step(
                V, reward_Q, reward_V, _Q,key,opt_state, (epoch+1) / 30 * i / 10
            )
            Q, q_opt_state, loss_value=Q_step(Q, V, q_opt_state)
            _Q=0.99*_Q+0.01*Q
        reward_V = jnp.zeros(gird_shape)
        reward_Q = -100*jnp.ones((*gird_shape, *gird_shape))
        _reward_Q=reward_Q
        for i in range(5000):
            reward_Q, reward_q_opt_state, loss_value = reward_Q_step(
            reward_Q, reward_V, V, reward_q_opt_state
            )
            reward_V, reward_opt_state, loss_value = reward_step(
                reward_V,_reward_Q, V, reward_opt_state
            )
            _reward_Q=0.99*_reward_Q+0.01*reward_Q

            if i  == 0:
                saveimg(reward_V, f"grid_img/test/0902/reward_V/{epoch}-{i}.jpg")
                saveimg(V, f"grid_img/test/0902/V/{epoch}-{i}.jpg")

        print(
            f"epoch: {epoch}, loss: {loss_value},reward_V_delta: {((old_reward_V-reward_V)**2).sum()}, V_delta: {((old_V-V)**2).sum()}"
        )
        old_reward_V = reward_V
        old_V = V

    return V


# print(res)
key = random.PRNGKey(758493)  # Random seed is explicit in JAX

# V=random.uniform(key, shape=gird_shape)
V = jnp.ones(gird_shape) * 0
optimizer = optax.adam(learning_rate=5e-2)
V = fit(V, optimizer)
print(V)
reward = np.zeros(gird_shape)
N = np.zeros(gird_shape)
stats = []
for traj in expert_trajs:
    stats.append((len(traj), calculate_y(traj, V).sum().item()))


for traj in sample_trajs:
    stats.append((len(traj), calculate_y(traj, V).sum().item()))
print(stats)
