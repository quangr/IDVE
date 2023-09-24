import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax 

@jax.jit
def large_function(x,_):
    callback={}
    def f1(x):
        callback["first"]=x+1
        return x+1
    def f2(x):
        callback["second"]=x**2
        return x**2
    return f1(f2(x)),callback



print(jax.lax.scan(large_function,1,None,length=10))