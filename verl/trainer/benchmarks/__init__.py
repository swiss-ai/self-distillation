from . import math, mcq
from .base import BenchmarkRunner


def build_benchmark_runners(config):
    return mcq.build_benchmark_runners(config) + math.build_benchmark_runners(config)


__all__ = [
    "build_benchmark_runners",
    "BenchmarkRunner",
    "math",
    "mcq",
]
