from dataclasses import dataclass
from ..core.tens.vec import Vector
from ..core.tens.mat import Matrix

@dataclass
class PipelineResult:
    original: Vector
    background: Vector| None
    foreground_mask: Vector| None
    def output():
        pass
