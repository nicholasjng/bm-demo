"""
JAX MNIST example with lakefs-spec.

This example demonstrates the use of lakefs-spec to train a ConvNet model
in a versioned manner in JAX/Flax on the MNIST dataset.

Source: https://github.com/google/flax/blob/main/examples/mnist
"""

import logging
import random
from pathlib import Path

import flax.linen as nn
import fsspec
import jax
import jax.numpy as jnp
import jax.random as jr
import lakefs
import numpy as np
import optax
from flax.training.train_state import TrainState

HERE = Path(__file__).parent

repo = lakefs.Repository("mnist").create(storage_namespace="local://mnist", exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ArrayMapping = dict[str, jax.Array | np.ndarray]

INPUT_SHAPE = (28, 28, 1)  # H x W x C (= 1, BW grayscale images)
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 0.1
MOMENTUM = 0.9


class ConvNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=NUM_CLASSES)(x)
        return x


@jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def create_train_state(rng):
    """Creates initial `TrainState`."""
    convnet = ConvNet()
    params = convnet.init(rng, jnp.ones([1, *INPUT_SHAPE]))["params"]
    tx = optax.sgd(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    return TrainState.create(apply_fn=convnet.apply, params=params, tx=tx)


def load_mnist() -> ArrayMapping:
    """
    Load MNIST dataset using fsspec.

    Returns
    -------
    ArrayMapping
        Versioned dataset as numpy arrays, split into training and test data.
    """

    if Path(HERE / "mnist.npz").exists():
        return np.load(HERE / "mnist.npz")

    mnist: ArrayMapping = {}

    baseurl = "http://yann.lecun.com/exdb/mnist/"

    for key, file in [
        ("x_train", "train-images-idx3-ubyte.gz"),
        ("x_test", "t10k-images-idx3-ubyte.gz"),
        ("y_train", "train-labels-idx1-ubyte.gz"),
        ("y_test", "t10k-labels-idx1-ubyte.gz"),
    ]:
        with fsspec.open(baseurl + file, compression="gzip") as f:
            if key.startswith("x"):
                mnist[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape((-1, 28, 28))
            else:
                mnist[key] = np.frombuffer(f.read(), np.uint8, offset=8)

    # save locally
    np.savez_compressed(HERE / "mnist.npz", **mnist)

    return mnist


def train_epoch(state, train_ds, train_labels, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds)
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds))
    # skip incomplete batch to avoid a recompile of apply_model
    perms = perms[: steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds[perm, ...]
        batch_labels = train_labels[perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def preprocess(data: ArrayMapping) -> ArrayMapping:
    """
    Expand dimensions of images and log sample images to MLflow.

    Args:
        data: ArrayMapping
            Raw input dataset, as a compressed NumPy array collection.
    Returns:
        ArrayMapping: Dataset with expanded dimensions.
    """

    data["x_train"] = jnp.float32(data["x_train"]) / 255.0
    data["y_train"] = jnp.float32(data["y_train"])
    data["x_test"] = jnp.float32(data["x_test"]) / 255.0
    data["y_test"] = jnp.float32(data["y_test"])

    # add a fake channel axis to make sure images have shape (28, 28, 1)
    if not data["x_train"].shape[-1] == 1:
        data["x_train"] = jnp.expand_dims(data["x_train"], -1)
        data["x_test"] = jnp.expand_dims(data["x_test"], -1)

    return data


def train(data: ArrayMapping) -> tuple[TrainState, ArrayMapping]:
    """Train a ConvNet model on the preprocessed data."""

    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    train_perm = np.random.permutation(len(x_train))
    train_perm = train_perm[: int(0.5 * len(x_train))]
    train_data, train_labels = x_train[train_perm, ...], y_train[train_perm, ...]

    test_perm = np.random.permutation(len(x_test))
    test_perm = test_perm[: int(0.5 * len(x_test))]
    test_data, test_labels = x_test[test_perm, ...], y_test[test_perm, ...]

    rng = jr.PRNGKey(random.randint(0, 1000))
    rng, init_rng = jr.split(rng)
    state = create_train_state(init_rng)

    for epoch in range(EPOCHS):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(
            state, train_data, train_labels, BATCH_SIZE, input_rng
        )
        # score the model on the test data
        _, test_loss, test_accuracy = apply_model(state, test_data, test_labels)

        logger.info(f"Logging metrics for epoch {epoch + 1}:")
        logger.info(f"Train loss: {train_loss.item()}")
        logger.info(f"Train accuracy: {train_accuracy.item() * 100.}")
        logger.info(f"Test loss: {test_loss.item()}")
        logger.info(f"Test accuracy: {test_accuracy.item() * 100.}")

    # the data we used in training.
    data = {
        "x_train": train_data,
        "y_train": train_labels,
        "x_test": test_data,
        "y_test": test_labels,
    }

    return state, data


def save(model: TrainState, data: ArrayMapping) -> None:
    """Score the model on the test data."""

    train_data = {
        "x_train": np.uint8(data["x_train"] * 255.),
        "y_train": np.uint8(data["y_train"]),
    }

    test_data = {
        "x_test": np.uint8(data["x_test"] * 255.),
        "y_test": np.uint8(data["y_test"]),
    }

    np.savez_compressed("train_data.npz", **train_data)
    np.savez_compressed("test_data.npz", **test_data)
    np.savez_compressed("model.npz", **model.params)

    logger.info("Saving train/test data split and model.")

    fs = fsspec.filesystem("lakefs")
    with fs.transaction as tx:
        # Upload the data...
        fs.put("train_data.npz", "mnist/experiment/train_data.npz")
        fs.put("test_data.npz", "mnist/experiment/test_data.npz")
        # ...and the model.
        fs.put("model.npz", "mnist/experiment/model.npz")
        tx.commit("mnist", "experiment", message="Add MNIST data and model")


def mnist_jax():
    """Load MNIST data and train a simple ConvNet model."""
    mnist = load_mnist()
    mnist = preprocess(mnist)
    model, data = train(mnist)
    save(model, data)


if __name__ == "__main__":
    mnist_jax()
