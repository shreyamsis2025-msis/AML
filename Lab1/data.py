import numpy as np

def generate_linear_data(n=100): 
    """A function that generates a linearly separable dataset"""
    X=np.random.randn(n,2)
    y=(X[:,0] + X[:,1] >0).astype(int)
    return X,y

def generate_xor_data(n=200):
    """A function that generates XOR type Data"""
    X=np.random.randn(n,2)
    y=((X[:,0]>0)^(X[:,1]>0)).astype(int)
    return X,y




# def generate_non_linear_data(n=100,noise=0.1):
#     radius=np.random.randn(n)
