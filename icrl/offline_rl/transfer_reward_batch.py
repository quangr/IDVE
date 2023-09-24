import argparse
import pickle
import time
import numpy as np
import os

import yaml
import random
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax.numpy as jnp
import jax
import optax
import matplotlib.pyplot as plt
from flax.training import orbax_utils
import orbax.checkpoint
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--expert-size", type=int, default=50,
        help="seed of the experiment")
    parser.add_argument("--lamb", type=float, default=0.4)
    parser.add_argument("--ad-min", type=float, default=float("-inf"))
    
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--setting", type=str, default="setting2",
        help="the wandb's project name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.setting}__{args.exp_name}__{args.seed}__{int(time.time()*1000)}"
    if args.track:
        import wandb

        wandb.init(
            project="grid_world",
            entity="quangr",
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

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


    def generate_random_path(grid, start, ends):
        def find_path(current_position, visited):
            if current_position in ends:
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


    def saveimg(data, save_path,title):
        fig, ax = plt.subplots()

        im = ax.imshow(data, cmap="gray", alpha=0.9)

        # Add a title
        ax.set_title(title)

        # Add labels to the x-axis and y-axis
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")

        # Add a colorbar
        fig.colorbar(im)
        for (j,i),label in np.ndenumerate(data):
            ax.text(i,j,"{:.2f}".format(label),ha='center',va='center')
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig(save_path)
        plt.close()

    grid_shape = (7, 7)
    settingname=args.setting
    expert_trajs = np.load(f"expert_data/grid_world/{settingname}/expert_trajs.npz")
    expert_trajs = [expert_trajs[name] for name in expert_trajs.files]
    if os.path.exists(f"expert_data/grid_world/{settingname}/expert_trajs_extra.npz"):
        expert_trajs_extra = np.load(f"expert_data/grid_world/{settingname}/expert_trajs_extra.npz")
        expert_trajs = expert_trajs+[expert_trajs_extra[name] for name in expert_trajs_extra.files]

    with open(f'expert_data/grid_world/{settingname}.yaml') as f:
        config=yaml.safe_load(f)


    start_point = tuple(expert_trajs[0][0].tolist())
    end_point = tuple(expert_trajs[0][-1].tolist())
    ab_points = list(map(tuple,config['terminal_states']))
    # expert_trajs = [traj[:6] for traj in expert_trajs]
    sample_trajs = []
    random_path = [
        generate_random_path(np.ones(grid_shape), start_point, ab_points)
        for i in range(50)
    ]
    sample_trajs += random_path
    random_path = [
        generate_random_path(np.ones(grid_shape), (1, 5), ab_points) for i in range(20)
    ]
    sample_trajs += random_path
    random_path = [
        generate_random_path(np.ones(grid_shape), (5, 1), ab_points) for i in range(20)
    ]
    sample_trajs += random_path
    random_path = [
        generate_random_path(np.ones(grid_shape), (1, 1), ab_points) for i in range(20)
    ]
    sample_trajs += random_path
    random_path = [
        generate_random_path(np.ones(grid_shape), (5, 5), ab_points) for i in range(20)
    ]
    sample_trajs += random_path

    d_ab = np.zeros(grid_shape)
    for point in ab_points:
        d_ab[point] = 1


    gamma = 0.99

    beta = 0.5

    rho_S = np.zeros((*grid_shape, *grid_shape))
    rho_E = np.zeros((*grid_shape, *grid_shape))

    for traj in expert_trajs:
        rho_E[traj[:-1, 0], traj[:-1:, 1], traj[1:, 0], traj[1:, 1]] += 1
    for traj in sample_trajs:
        rho_S[traj[:-1, 0], traj[:-1:, 1], traj[1:, 0], traj[1:, 1]] += 1

    state_ocuupy_E = np.zeros(grid_shape)
    for traj in expert_trajs:
        state_ocuupy_E[traj[:-1, 0], traj[:-1, 1]] += 1


    rho = (beta * rho_E + (1 - beta) * rho_S) / (
    (beta * rho_E + (1 - beta) * rho_S).sum((-1, -2))
    ).clip(0.1)[..., None, None]
    expert_d = rho_E / rho_E.sum((-1, -2)).clip(1)[..., None, None]
    expert_d[(expert_d<1)&(expert_d>0)]=0.5

    def get_data(grid_shape, expert_trajs, end_point, sample_trajs, gamma, beta):
        d_end = np.zeros(grid_shape)
        d_end[end_point] = 1

        def calculate_traj_distances(traj):
            r = (traj[1:] == end_point).all(axis=1) - 1

            d = np.power(gamma, np.arange(len(traj) - 1))
            return d, r

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
        return target_E,target_S,show_S,r_E,r_S,obs_E,obs_next_E,obs_S,obs_next_S,d_end




    # def f_div(x):
    #     return x*jnp.log(x)

    # def omega_star(y):
    #     return jnp.exp(y-1)

    def f_div(x):
        return (x-1)**2

    def omega_star(y):
        return (y / 2 + 1).clip(0)

    def fp_star(y):
        return omega_star(y) * y - f_div(omega_star(y))



    cost_max=0
    cost_min=-jnp.inf



    def solve_V(grid_shape, expert_trajs, end_point, sample_trajs, gamma, beta,lamb,ad_min,expert_indice):
        target_E,target_S,show_S,r_E,r_S,obs_E,obs_next_E,obs_S,obs_next_S,d_end = get_data(grid_shape, expert_trajs, end_point, sample_trajs, gamma, beta)
        obs_E=obs_E[expert_indice]
        obs_next_E=obs_next_E[expert_indice]
        r_E=r_E[expert_indice]

        def calculate_y(obs, obs_next, V):
            y = gamma * V[obs_next[:, 0], obs_next[:, 1]] - V[obs[:, 0], obs[:, 1]]
            return y,  V[obs[:, 0], obs[:, 1]],V[obs_next[:, 0], obs_next[:, 1]]

        def get_V(V):
            y_E, V_E, next_V_E = calculate_y(obs_E, obs_next_E, V)
            y_S, V_S, next_V_S = calculate_y(obs_S, obs_next_S, V)
            return (y_E, V_E, next_V_E), (y_S, V_S, next_V_S)
        @jax.jit
        def dualrewardV(reward_V, V):

            (y_E, V_E, next_V_E), (y_S, V_S, next_V_S) = get_V(reward_V)
            (cost_y_E, _, _), (cost_y_S, _, _) = get_V(V)
            cost_y_E = cost_y_E.clip(cost_min, cost_max)
            cost_y_S = cost_y_S.clip(cost_min, cost_max)
            return (
                (1 - lamb) * (beta * (V_E-gamma*next_V_E).mean() + (1 - beta) * (V_S-gamma*next_V_S).mean())
                + lamb
                * (
                    beta * (fp_star(y_E + r_E + cost_y_E) ).mean()
                    + (1 - beta) * (fp_star(y_S + r_S + cost_y_S)).mean()
                )
                + 10 * ((reward_V * (d_end))**2).sum()
                + 10 * (((reward_V+1/(1-gamma)) * (d_ab-d_end))**2).sum()
            )

        p_lower=0.005
        @jax.jit
        def dualV(V, reward_V,V_alpha=1):

            (y_E, V_E, next_V_E), (y_S, V_S, next_V_S) = get_V(reward_V)
            # V=V.clip(max=0)
            (cost_y_E, cost_V_E, next_cost_V_E), (cost_y_S, cost_V_S, next_cost_V_S) = get_V(V)
            pi_E = omega_star(y_E + r_E + cost_y_E.clip(cost_min, cost_max))
            pi_S = omega_star(y_S + r_S + cost_y_S.clip(cost_min, cost_max))
            cost = 1 * (
                (1 - beta)
                * (
                        jnp.abs(cost_y_S)
                        + jnp.exp(cost_y_S.clip(cost_max) - cost_y_S.clip(max=cost_min))
                ).mean()
                + (beta)
                * (
                        jnp.abs(cost_y_E)
                        + jnp.exp(50 * cost_y_E.clip(cost_max) - cost_y_E.clip(max=cost_min))
                    
                ).mean()
            ) + V_alpha *5* (
                (1 - beta) * (jnp.abs(next_cost_V_S) + cost_V_S.clip(0)).mean()
                + (beta) * (jnp.abs(next_cost_V_E) + cost_V_E.clip(0)).mean()
            )
            return (
                beta * -((y_E + r_E + cost_y_E.clip(cost_min, cost_max))).mean()
                +(1-beta)*(show_S*(target_S==0)*(y_S + r_S + cost_y_S.clip(cost_min, cost_max)).clip(ad_min)).mean()
                # + cost*0.00001
                + jnp.abs(cost_y_E).mean()*10
                + 10 * (jnp.abs((V * d_ab))).sum()
            )



        def fit(optimizer):
            V = jnp.zeros(grid_shape)
            reward_V = jnp.zeros(grid_shape)
            alpha = jnp.zeros(1)
            opt_state = optimizer.init(V)
            reward_opt_state = optimizer.init(reward_V)

            @jax.jit
            def reward_step(V, reward_V, opt_state):
                loss_value, grads = jax.value_and_grad(dualrewardV)(reward_V, V)
                updates, opt_state = optimizer.update(grads, opt_state, reward_V)
                reward_V = optax.apply_updates(reward_V, updates)
                return reward_V, opt_state, loss_value


            @jax.jit
            def step(V, reward_V, opt_state,V_alpha):
                loss_value, grads = jax.value_and_grad(dualV)(V, reward_V,V_alpha.clip(max=700))
                updates, opt_state = optimizer.update(grads, opt_state, V)
                V = optax.apply_updates(V, updates)
                return V, opt_state, loss_value

            is_reward = 0
            
            reward_V = jnp.zeros(grid_shape)
            for i in range(50000):
                reward_V, reward_opt_state, loss_value = reward_step(
                    V, reward_V, reward_opt_state
                )
            old_reward_V=reward_V
            old_V=V
            for epoch in range(10):
                opt_state = optimizer.init(V)
                reward_opt_state = optimizer.init(reward_V)
                # saveimg(reward_V,f"grid_img/test/reward_V/{epoch}.jpg","Reward V function")
                # saveimg(V,f"grid_img/test/V/{epoch}.jpg","Recovered Cost V function")
                # saveimg(reward_V+V,f"grid_img/test/total_V/{epoch}.jpg","Recovered Total V function")
                V = jnp.zeros(grid_shape)
                for i in range(20000):
                    V, opt_state, loss_value = step(V, reward_V, opt_state,epoch/30*i/10)
                # reward_V = jnp.zeros(grid_shape)
                for i in range(10000):
                    reward_V, reward_opt_state, loss_value = reward_step(
                        V, reward_V, reward_opt_state
                    )

                    # if i % 1000 == 0:
                print(
                    f"epoch: {epoch}, loss: {loss_value},reward_V_delta: {((old_reward_V-reward_V)**2).sum()}, V_delta: {((old_V-V)**2).sum()}"
                )               
                old_reward_V=reward_V
                old_V=V                


            return reward_V,V


        # print(res)
        key = jax.random.PRNGKey(758493)  # Random seed is explicit in JAX

        # V=random.uniform(key, shape=grid_shape)
        optimizer = optax.adam(learning_rate=1e-2)
        reward_V,V = fit(optimizer)
        return reward_V,V


    def resolve_V(grid_shape, expert_trajs, end_point, sample_trajs, gamma, beta,V,expert_indice):
        target_E,target_S,show_S,r_E,r_S,obs_E,obs_next_E,obs_S,obs_next_S,d_end = get_data(grid_shape, expert_trajs, end_point, sample_trajs, gamma, beta)

        obs_E=obs_E[expert_indice]
        obs_next_E=obs_next_E[expert_indice]
        r_E=r_E[expert_indice]

        def calculate_y(obs, obs_next, V):
            y = gamma * V[obs_next[:, 0], obs_next[:, 1]] - V[obs[:, 0], obs[:, 1]]
            return y,  V[obs[:, 0], obs[:, 1]],V[obs_next[:, 0], obs_next[:, 1]]

        def get_V(V):
            y_E, V_E, next_V_E = calculate_y(obs_E, obs_next_E, V)
            y_S, V_S, next_V_S = calculate_y(obs_S, obs_next_S, V)
            return (y_E, V_E, next_V_E), (y_S, V_S, next_V_S)
        @jax.jit
        def dualrewardV(reward_V, V):

            (y_E, V_E, next_V_E), (y_S, V_S, next_V_S) = get_V(reward_V)
            (cost_y_E, _, _), (cost_y_S, _, _) = get_V(V)
            cost_y_E = cost_y_E.clip(cost_min, cost_max)
            cost_y_S = cost_y_S.clip(cost_min, cost_max)
            return (
                (1 - lamb) * (beta * (V_E-gamma*next_V_E).mean() + (1 - beta) * (V_S-gamma*next_V_S).mean())
                + lamb
                * (
                    beta * (fp_star(y_E + r_E + cost_y_E) ).mean()
                    + (1 - beta) * (fp_star(y_S + r_S + cost_y_S)).mean()
                )
                + 10 * ((reward_V * (d_end))**2).sum()
                + 10 * (((reward_V+1/(1-gamma)) * (d_ab-d_end))**2).sum()
            )


        def fit(V,optimizer):
            reward_V = jnp.zeros(grid_shape)
            reward_opt_state = optimizer.init(reward_V)

            @jax.jit
            def reward_step(V, reward_V, opt_state):
                loss_value, grads = jax.value_and_grad(dualrewardV)(reward_V, V)
                updates, opt_state = optimizer.update(grads, opt_state, reward_V)
                reward_V = optax.apply_updates(reward_V, updates)
                return reward_V, opt_state, loss_value


            reward_V = jnp.zeros(grid_shape)
            for i in range(50000):
                reward_V, reward_opt_state, loss_value = reward_step(
                    V, reward_V, reward_opt_state
                )
            old_reward_V=reward_V
            for epoch in range(1):
                reward_opt_state = optimizer.init(reward_V)
                # saveimg(reward_V,f"grid_img/test/reward_V/{epoch}.jpg")
                # saveimg(reward_V+V,f"grid_img/test/total_V/{epoch}.jpg")
                V = jnp.zeros(grid_shape)
                for i in range(10000):
                    reward_V, reward_opt_state, loss_value = reward_step(
                        V, reward_V, reward_opt_state
                    )

                print(
                    f"epoch: {epoch}, loss: {loss_value},reward_V_delta: {((old_reward_V-reward_V)**2).sum()}"
                )               
                old_reward_V=reward_V


            return reward_V


        # print(res)
        key = jax.random.PRNGKey(758493)  # Random seed is explicit in JAX

        # V=random.uniform(key, shape=grid_shape)
        optimizer = optax.adam(learning_rate=1e-2)
        reward_V = fit(V,optimizer)
        return reward_V





    def get_traj(V, end_point, reward_V):
        cum_cost = np.zeros(grid_shape) 
        cum_length = np.zeros(grid_shape) 
        direction=np.zeros((*grid_shape,2)) 

        def generate_trajectory(direction,start,end):
            state=start
            traj_set=[state]
            for i in range(50):
                if state!=end:
                    new_state=tuple((state+direction[state]).tolist())
                    traj_set.append(new_state)
                    state=new_state
                else:
                    return traj_set
            return traj_set
        


        for x in range(grid_shape[0]):
            for y in range(grid_shape[1]):
                if (x,y) not in ab_points:
                    action=rho[x,y]>0
                    maxaction=np.flatnonzero(action)[np.argmax((reward_V+V)[action])]
                    new_state=np.unravel_index(maxaction, grid_shape)
                    direction[x,y]=(new_state[0]-x,new_state[1]-y)
        direction=direction.astype(np.int32)
        for x in range(grid_shape[0]):
            for y in range(grid_shape[1]):
                if (x,y) not in ab_points:
                    traj=generate_trajectory(direction,(x,y),end_point)
                    if traj[-1] !=end_point:
                        cum_length[x,y]=np.nan
                    else:
                        cum_length[x,y]=len(traj)-1
                    cum_cost[x,y]=len(set(map(tuple,config['unsafe_states'])).intersection(traj))
        print(cum_cost)
        print(cum_length)
        return cum_cost,cum_length

    lamb = args.lamb
    ad_min= args.ad_min
    costs=[]
    lengths=[]
    Vs=[]
    for i in range(5):
        expert_indice = np.random.permutation((np.array(list(map(len,expert_trajs)))-1).sum())[:args.expert_size]
        reward_V,V=solve_V(grid_shape, expert_trajs, end_point, sample_trajs, gamma, beta,lamb,ad_min,expert_indice)
        new_end_point=ab_points[-1]
        new_reward_V=resolve_V(grid_shape, expert_trajs, new_end_point, sample_trajs, gamma, beta,V,expert_indice)
        cum_cost,cum_length=get_traj(V, new_end_point, new_reward_V)
        alter_start = list(map(tuple,config['alter_start_states']))

        d_start = np.zeros(grid_shape,dtype=bool)
        for point in alter_start:
            d_start[point] = True
        Vs.append((reward_V,V,new_reward_V))
        costs.append(cum_cost[d_start].mean())
        lengths.append(cum_length[d_start].mean())
    if args.track:
        wandb.run.summary["mean_cost"]=np.array(costs).mean()
        wandb.run.summary["violate rate"]=np.array(costs>0).mean()
        wandb.run.summary["mean_length"]=np.array(lengths).mean()
        wandb.run.summary["successful rate"]=1-np.isnan(np.array(lengths)).mean()
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(Vs)
        checkpointer.save(f"grid_img/paper/wandb/{run_name}",Vs, save_args=save_args)


    # new_reward_V=resolve_V(grid_shape, expert_trajs, new_end_point, sample_trajs, gamma, beta,V)
    # get_traj(V, new_end_point, new_reward_V)


    # checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # save_args = orbax_utils.save_args_from_target((V,reward_V,new_reward_V))
    # checkpointer.save(f"grid_img/paper/{settingname}",(V,reward_V,new_reward_V), save_args=save_args)

    # V=jnp.zeros(grid_shape)
    # new_reward_V=resolve_V(grid_shape, expert_trajs, new_end_point, sample_trajs, gamma, beta,V)
    # get_traj(V, new_end_point, new_reward_V)
