import numpy as np
def make_client_data(n_clients=20, n_features=10, seed=42):
    rng=np.random.default_rng(seed); out=[]
    for c in range(n_clients):
        shift=rng.normal(0,1,size=n_features)*(0.5+0.5*rng.random()); n=rng.integers(800,1400)
        X=rng.normal(0,1,size=(n,n_features))+shift; w=rng.normal(0,1,size=n_features)
        score=X@w + rng.normal(0,0.5,size=n); y=(score>np.percentile(score,90)).astype(int); out.append((X,y))
    return out
