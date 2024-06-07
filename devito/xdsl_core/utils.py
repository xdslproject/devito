
from typing import Iterable


def generate_pipeline(passes: Iterable[str]):
    'Generate a pipeline string from a list of passes'
    passes_string = ",".join(passes)
    return f'"{passes_string}"'


def generate_mlir_pipeline(passes: Iterable[str]):
    passes_string = ",".join(passes)
    return f'mlir-opt[{passes_string}]'
