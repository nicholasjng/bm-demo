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
    version = kwargs['version']
    rpath_prefix = f"mnist/{version}/"

    # load test datasets
    data = kwargs["data"]
    fs.get(rpath_prefix + data, data)
    datadict = np.load(data)

    # load model parameters
    paramfile = kwargs["params"]
    fs.get(rpath_prefix + paramfile, paramfile)
    params = {k: v.item() for k, v in np.load(paramfile, allow_pickle=True).items()}

    return {
        "params": params,
        "data": datadict,
        "version": version,
    }


def tearDown(**kwargs):
    Path(kwargs["data"]).unlink(missing_ok=True)
    Path(kwargs["params"]).unlink(missing_ok=True)


@parametrize(
    (dict(params="model.npz", data="test_data.npz", version=f"v{n}") for n in range(1, 6)),
    setUp=setUp,
    tearDown=tearDown,
)
def accuracy(params: ArrayMapping, data: ArrayMapping, version: str) -> float:
    x_test, y_test = data["x_test"], data["y_test"]
    x_test = jnp.float32(x_test) / 255.0

    cn = ConvNet()
    y_pred = cn.apply({"params": params}, x_test)
    return jnp.mean(jnp.argmax(y_pred, -1) == y_test).item()


if __name__ == '__main__':
    BenchmarkRunner().run()
