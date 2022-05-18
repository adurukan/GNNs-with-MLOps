# GNNs-with-MLOps
Various GNN algorithms such as GAT and GCN are trained on datasets to detect the pre-determined patterns. Overall, running an end-to-end procedure with assistance of complete MLOps for node-classification tasks.

# LOCAL TESTING

I would suggest creating a virtual environment using venv, the way to do is: 
- python3 -m venv .venv/
- source .venv/bin/activate

Later, please install the requirements-pip.txt using 
- pip install -r requirements-pip.txt

In case of issues regarding torch, torchvision or torchaudio please use
- Remove them either with pip uninstall or simply remove from requirements-pip.txt
- pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

In case of issues regarding torch-scatter, torch-sparse or torch-geometric please use 
- Remove them either with pip uninstall or simply remove from requirements-pip.txt
- pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
- pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
- pip install torch-geometric


