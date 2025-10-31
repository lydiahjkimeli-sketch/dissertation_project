# FL-DP Vehicular Anomaly Detection (Dissertation Code)

Minimal, runnable reference for a hybrid **Federated Learning + Differential Privacy** anomaly detection workflow in vehicular networks.

See `train_federated.py` for a complete CPU-friendly simulation.

## Quickstart
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_federated.py --clients 10 --rounds 20 --epsilon 0.5 --noise-multiplier 0.8 --clip-norm 1.0
# dissertation_project
# dissertation_project
