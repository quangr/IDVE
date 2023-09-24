env_id="Ant-v3"

def cost_function(next_obs, reward, next_done, next_truncated, info):
    if 'true_obs' in info.keys():
        return  (info['true_obs'][:,0]>0.45)
    else:
        return (next_obs[:,0]>0.45)
