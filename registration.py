from dataclasses import dataclass
from typing import Iterable, Callable, Any


@dataclass(frozen=True)
class Benchmarkable:
    fn: Callable
    name: str
    params: dict[str, Any]
    setUp: Callable
    tearDown: Callable


def parametrize(argiter: Iterable[dict], setUp: Callable, tearDown: Callable) -> Callable:
    def inner(fn: Callable) -> list[Benchmarkable]:
        benchmark_list = []
        for params in argiter:
            name = fn.__name__ + "_" + "_".join(f"{k}={v}" for k, v in params.items())
            bm = Benchmarkable(
                fn, name=name, params=params, setUp=setUp, tearDown=tearDown,
            )
            benchmark_list.append(bm)

        return benchmark_list
    return inner
