import sys
from dataclasses import dataclass

from tabulate import tabulate
from typing import Any, Callable

from registration import Benchmarkable


@dataclass(frozen=True)
class Benchmark:
    fn: Callable
    name: str
    params: dict[str, Any]
    setUp: Callable
    tearDown: Callable


def is_dunder(s: str) -> bool:
    return s.startswith("__") and s.endswith("__")


class BenchmarkRunner:

    benchmark_type = Benchmark

    def __init__(self):
        self.benchmarks = None

    def discover(self, module: str | None = None) -> list[Benchmark]:
        if module and module != "__main__":
            raise NotImplementedError

        module = module or "__main__"
        moddict = sys.modules[module].__dict__

        filtered = {k: moddict[k] for k in filter(lambda k: not is_dunder(k), moddict)}
        benchmarkables = []
        for k, v in filtered.items():
            if isinstance(v, Benchmarkable):
                benchmarkables.append(v)
            elif isinstance(v, list) and all(isinstance(b, Benchmarkable) for b in v):
                benchmarkables.extend(v)

        benchmarks = [
            self.benchmark_type(
                fn=b.fn, name=b.name, params=b.params, setUp=b.setUp, tearDown=b.tearDown,
            ) for b in benchmarkables
        ]
        # memoize
        self.benchmarks = benchmarks
        if not benchmarks:
            raise RuntimeError("no benchmarks found")
        return benchmarks

    def run(self) -> None:
        self.discover()
        results = []
        for benchmark in self.benchmarks:
            res = self.run_benchmark(benchmark)
            results.append(res)

        print(tabulate(results, headers="keys"))
        self.benchmarks = None

    def run_benchmark(self, benchmark: Benchmark) -> dict[str, Any]:
        pkwargs = benchmark.setUp(**benchmark.params)
        accuracy = benchmark.fn(**pkwargs)
        benchmark.tearDown(**benchmark.params)

        return {
            "name": benchmark.name,
            "accuracy": accuracy,
            "version": benchmark.params["version"]
        }
