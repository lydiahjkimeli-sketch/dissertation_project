import argparse, numpy as np, pandas as pd, os
from fl_dp.federated import Client, ClientConfig, fedavg, evaluate
from data.make_synthetic import make_client_data
from evaluate import classification_metrics
def args():
    ap=argparse.ArgumentParser(); ap.add_argument('--clients',type=int,default=10); ap.add_argument('--rounds',type=int,default=20)
    ap.add_argument('--local-epochs',type=int,default=3); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--clip-norm',type=float,default=1.0); ap.add_argument('--noise-multiplier',type=float,default=0.8); ap.add_argument('--epsilon',type=float,default=0.5)
    ap.add_argument('--seed',type=int,default=42); return ap.parse_args()
def main():
    a=args(); rng=np.random.default_rng(a.seed)
    clients_data=make_client_data(n_clients=a.clients, n_features=10, seed=a.seed); in_dim=clients_data[0][0].shape[1]
    X_test=np.vstack([c[0][:200] for c in clients_data]); y_test=np.hstack([c[1][:200] for c in clients_data])
    cfg=ClientConfig(lr=a.lr, local_epochs=a.local_epochs, batch_size=a.batch_size, clip_norm=a.clip_norm, noise_multiplier=a.noise_multiplier)
    clients=[Client(i,X,y,in_dim,cfg=cfg) for i,(X,y) in enumerate(clients_data)]
    global_state=clients[0].get_weights(); hist=[]
    for r in range(a.rounds):
        chosen=rng.choice(len(clients), size=max(1,len(clients)//2), replace=False); updates=[]
        for idx in chosen:
            clients[idx].set_weights(global_state); updates.append(clients[idx].local_train())
        global_state=fedavg(updates)
        y_prob=evaluate(global_state,X_test,y_test,in_dim); m=classification_metrics(y_test,y_prob,0.5)
        hist.append(dict(round=r+1, epsilon=a.epsilon, clients=int(len(chosen)), **m))
        print(f"round {r+1:03d} eps={a.epsilon} clients={len(chosen)} f1={m['f1']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} auc={m['auc']:.3f}")
    os.makedirs('figures',exist_ok=True); pd.DataFrame(hist).to_csv('results.csv',index=False); print("saved results.csv")
if __name__=='__main__': main()
