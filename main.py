from pathlib import Path

import jax
import numpy as np
import jax.numpy as jnp

from lakefs_spec import LakeFSFileSystem

from mnist import ConvNet
from registration import parametrize
from runner import BenchmarkRunner

ArrayMapping = dict[str, jax.Array | np.ndarray]


def setUp(**kwargs):
    fs = LakeFSFileSystem()

    # load test dataset
    rpath = kwargs["data"]
    lpath = "test_data.npz"
    fs.get(rpath, lpath)
    datadict = np.load(lpath)

    # load model parameters
    param_rpath = kwargs["params"]
    param_lpath = "model.npz"
    fs.get(param_rpath, param_lpath)
    params = {k: v.item() for k, v in np.load(param_lpath, allow_pickle=True).items()}

    return {
        "data": datadict,
        "params": params,
    }


def tearDown(**kwargs):
    npzfiles = [f for f in Path.cwd().iterdir() if f.suffix == ".npz"]
    for file in npzfiles:
        file.unlink(missing_ok=True)


@parametrize(
    (dict(params=f"mnist/v{n}/model.npz", data=f"mnist/v{n}/test_data.npz") for n in range(1, 6)),
    setUp=setUp,
    tearDown=tearDown,
)
def accuracy(params: ArrayMapping, data: ArrayMapping) -> float:
    x_test, y_test = data["x_test"], data["y_test"]
    x_test = jnp.float32(x_test) / 255.0

    cn = ConvNet()
    y_pred = cn.apply({"params": params}, x_test)
    return jnp.mean(jnp.argmax(y_pred, -1) == y_test).item()


if __name__ == '__main__':
    BenchmarkRunner().run()
