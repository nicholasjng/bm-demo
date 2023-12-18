# A ⚡️demo on AI model benchmarking

This repo is a demo on an upcoming MLOps team project (current name `nnbench`).

It demonstrates a set of abstractions related to easy, reproducible, and customizable benchmarking of AI models on different tasks.

## Setup & prerequisites

After cloning this repository, run the following commands to set up the environment:

```shell
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install "jax[cpu]" flax optax tensorflow-datasets tensorflow tabulate lakefs-spec
```

Additionally, set up a local lakeFS instance via `Docker`, and place a configuration file in your home directory to authenticate with it:

```shell
docker run --name lakefs --pull always \
             --rm --publish 8000:8000 \
             treeverse/lakefs:latest \
             run --quickstart

cp .lakectl.yaml "$HOME/.lakectl.yaml"
```

## Training models and benchmarking them

After completing the above setup, run the `mnist.py` script to train five MNIST models and push them to lakeFS as distinct versions.
These versions are each given a _branch_, `v${N}` (for the number of the model), starting at 1.

```shell
python mnist.py
```

After running this script (taking about 2 minutes on an Apple Silicon Macbook), you may run the `main.py` script to benchmark each of them for accuracy on their respective test folds:

```shell
python main.py
```

This should result in a tabular output of benchmark names, accuracy, and model version, like the one below.

```shell
➜ python main.py
name                                                              accuracy  version
--------------------------------------------------------------  ----------  ---------
accuracy_params=mnist/v1/model.npz_data=mnist/v1/test_data.npz      0.9694  v1
accuracy_params=mnist/v2/model.npz_data=mnist/v2/test_data.npz      0.9726  v2
accuracy_params=mnist/v3/model.npz_data=mnist/v3/test_data.npz      0.9776  v3
accuracy_params=mnist/v4/model.npz_data=mnist/v4/test_data.npz      0.9768  v4
accuracy_params=mnist/v5/model.npz_data=mnist/v5/test_data.npz      0.9722  v5
```

