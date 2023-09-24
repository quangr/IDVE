import numpy as np
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax.numpy as jnp
import jax 
import optax
from jax import random
def create_coordinate_set(x1, y1, x2, y2):
  if x1 == x2:
    coords = [(x1, i) for i in range(y1,y2+(-1 if y1 > y2 else 1), -1 if y1 > y2 else 1)]
  elif y1 == y2:
    coords = [(i, y1) for i in range(x1, x2+(-1 if x1 > x2 else 1), -1 if x1 > x2 else 1)]
  return np.array(coords)

gird_shape=(2,2)
# expert_trajs=np.load("expert_data/grid_world/setting1/expert_trajs.npz")
# expert_trajs=[expert_trajs[name] for name in expert_trajs.files]
expert_trajs=[np.array([(0,0),(0,1)])]
sample_trajs=[np.array([(0,0),(1,0),(1,1),(0,1)])]
# sample_trajs=[]
# sample_trajs_next=[]
# for x in range(7):
#     for y in range(7):
#        adj=list(filter(lambda cord: cord[0]>=0 and cord[1]>=0 and cord[0]<7 and cord[1]<7,[(x+1,y), (x-1,y), (x,y+1), (x,y-1), (x+1,y+1), (x+1,y-1), (x-1,y+1), (x-1,y-1)]))
#        sample_trajs+=[(x,y)]*len(adj)
#        sample_trajs_next+=adj



start_point=(0,0)
# end_point=(5,0)
d_0=np.zeros(gird_shape)
d_0[start_point]=1



gamma=0.99

beta=0.5

def calculate_traj_distances(traj):
    r = np.zeros(len(traj) - 1)
    r[:-1] = -1
    d = np.power(gamma, np.arange(len(traj) - 1))
    return d,r

def calculate_y(traj,V):
    y = gamma * V[traj[1:, 0], traj[1:, 1]] - V[traj[:-1, 0], traj[:-1, 1]]
    return y

# def fp_star(y,rho):
#     return omega_star(y,rho)*y-(omega_star(y,rho)-1)**2
def fp_star(y,rho):
    return (rho>0)*((y>=1/2-0.1)*y+(y<=-3/2)*(-y)+(-2*jnp.sqrt(2)*(1/2-y)-(y-1/2)**2)*(y<1/2-0.1)*(y>-3/2))+(rho<=0)*(y.clip(1/2))

def omega_star(y,rho):
    return (y<0)*rho*(jnp.sqrt(1-2*y)-1)
rho_S=np.zeros((*gird_shape,*gird_shape))
rho_E=np.zeros((*gird_shape,*gird_shape))

for traj in expert_trajs:
   rho_E[traj[:-1, 0], traj[:-1, 1],traj[1:, 0], traj[1:, 1]]+=1
for traj in sample_trajs:
   rho_S[traj[:-1, 0], traj[:-1, 1],traj[1:, 0], traj[1:, 1]]+=1
# for x,y in sample_trajs_next:
#    rho_S[x, y]+=1

rho_E[rho_E==0]=1/4

rho=rho_E/((beta*rho_E+(1-beta)*rho_S).clip(.01))

def get_all_data(trajs, rho):
    d_E,r_E=list(zip(*[calculate_traj_distances(traj) for traj in trajs]))
    d_E=np.concatenate(d_E)
    r_E=np.concatenate(r_E)
    rho_E=np.concatenate([rho[traj[:-1, 0], traj[:-1, 1],traj[1:, 0], traj[1:, 1]] for traj in trajs])
    return rho_E,d_E,r_E


rho_E, d_E, r_E = get_all_data(expert_trajs, rho)
rho_S, d_S, r_S = get_all_data(sample_trajs, rho)

# sample_trajs=np.array(sample_trajs)
# sample_trajs_next=np.array(sample_trajs_next)
# d_S,r_S=np.ones(len(sample_trajs)),(((sample_trajs[:,0]==6) & (sample_trajs[:,1]==0))-1)
# rho_S=rho[sample_trajs_next[:,0], sample_trajs_next[:,1]]

def get_V(V):
    y_E=jnp.concatenate([calculate_y(traj,V) for traj in expert_trajs])
    y_S=jnp.concatenate([calculate_y(traj,V) for traj in sample_trajs])
    return y_E,y_S #+ 10*((1-beta)*(d_S*jnp.abs(y_S)).mean() +(beta)*(d_E*jnp.abs(y_E)).mean())

@jax.jit
def dualV(V,alpha=0):   
    y_E, y_S = get_V(V)
    return (1 - gamma) * beta * (V * d_0).sum() + beta * (fp_star(y_E,rho_E) * d_E).mean() + (1 - beta) * (fp_star(y_S,rho_S) * d_S).mean() 


@jax.jit
def CdualV(V,alpha):
    def fp_star(y,rho):
        return (rho>0)*((y>=1/2-0.1)*y+(y<=-3/2)*(-y)+(-2*jnp.sqrt(2)*(1/2-y)-(y-1/2)**2)*(y<1/2-0.1)*(y>-3/2))+(rho<=0)*(y.clip(1/2))
    y_E, y_S = get_V(V)
    return (1 - gamma) * beta * (V * d_0).sum() + beta * (fp_star(y_E+alpha*r_E,rho_E) * d_E).mean() + (1 - beta) * (fp_star(y_S+alpha*r_E,rho_S) * d_S).mean() 


@jax.jit
def solvealpha(alpha,V):
    y_E, y_S = get_V(V)
    return (beta * (d_E * (omega_star(-alpha*r_E+y_E,rho_E) - rho_E) * r_E).mean() + (1 - beta) * (d_S * (omega_star(-alpha*r_S+y_S,rho_S) - rho_S) * r_S).mean())**2

def dualV_constraint(V,alpha):    
    y_E, y_S = get_V(V)
    is_reward=beta * (d_E * (omega_star(y_E,rho_E) - rho_E) * r_E).mean() + (1 - beta) * (d_S * (omega_star(y_S,rho_S) - rho_S) * r_S).mean()
    return jax.lax.cond(is_reward>0,dualV,CdualV,V,alpha),is_reward


def fit(V, optimizer):
  alpha=jnp.zeros(1)
  opt_state = optimizer.init(V)
  alpha_opt_state=optimizer.init(alpha)
  is_reward=0
  @jax.jit
  def step(V, opt_state,alpha):
    (loss_value,is_reward), grads = jax.value_and_grad(dualV_constraint,has_aux=True)(V,alpha)
    # loss_value, grads = jax.value_and_grad(dualV)(V,alpha)
    updates, opt_state = optimizer.update(grads, opt_state, V)
    V = optax.apply_updates(V, updates)
    return V, opt_state, loss_value,is_reward
  @jax.jit
  def step_alpha(V, opt_state,alpha):
    loss_value, grads = jax.value_and_grad(solvealpha)(alpha,V)
    updates, opt_state = optimizer.update(grads, opt_state, alpha)
    alpha = optax.apply_updates(alpha, updates)
    return alpha, opt_state, loss_value

  for i in range(10000):
    alpha, alpha_opt_state, loss_value = step_alpha(V, alpha_opt_state,alpha)
    alpha=alpha.clip(0)
    # alpha=0.
    V, opt_state, loss_value,is_reward = step(V, opt_state,alpha)    
    if i % 1000 == 0:
      print(f'step {i}, loss: {loss_value},alpha: {alpha}, is_reward: {is_reward}')

  return V
# print(res)
key = random.PRNGKey(758493)  # Random seed is explicit in JAX

# V=random.uniform(key, shape=gird_shape)
V=jnp.ones(gird_shape)*(-1)
optimizer = optax.adam(learning_rate=5e-3)
V = fit(V, optimizer)
print(V)
reward=np.zeros(gird_shape)
N=np.zeros(gird_shape)
stats=[]
for traj in expert_trajs:
    stats.append((len(traj),calculate_y(traj,V).sum().item()))

sample_trajs=np.load("expert_data/grid_world/setting1/sample_trajs.npz")

sample_trajs=[sample_trajs[name] for name in sample_trajs.files]

for traj in sample_trajs:
    stats.append((len(traj),calculate_y(traj,V).sum().item()))
print(stats)