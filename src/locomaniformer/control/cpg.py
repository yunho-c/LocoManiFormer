from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CPGParameters:
    amplitudes: FloatArray
    offsets: FloatArray
    frequencies: FloatArray
    phase_biases: FloatArray
    coupling_weights: FloatArray

    def copy(self) -> CPGParameters:
        return CPGParameters(
            amplitudes=self.amplitudes.copy(),
            offsets=self.offsets.copy(),
            frequencies=self.frequencies.copy(),
            phase_biases=self.phase_biases.copy(),
            coupling_weights=self.coupling_weights.copy(),
        )

    def to_dict(self) -> dict[str, list[float] | list[list[float]]]:
        return {
            "amplitudes": self.amplitudes.tolist(),
            "offsets": self.offsets.tolist(),
            "frequencies": self.frequencies.tolist(),
            "phase_biases": self.phase_biases.tolist(),
            "coupling_weights": self.coupling_weights.tolist(),
        }


@dataclass(frozen=True)
class CPGState:
    time: float
    phases: FloatArray
    outputs: FloatArray


class CPG:
    def __init__(self, parameters: CPGParameters, dt: float) -> None:
        self.parameters = parameters
        self.dt = dt

    def reset(self, rng: np.random.Generator | None = None) -> CPGState:
        phases = np.zeros_like(self.parameters.amplitudes, dtype=np.float64)
        if rng is not None and phases.size:
            phases += rng.uniform(-1e-3, 1e-3, size=phases.shape)
        outputs = self._outputs(phases)
        return CPGState(time=0.0, phases=phases, outputs=outputs)

    def step(self, state: CPGState) -> CPGState:
        phase_deltas = self._phase_derivative(state.phases)
        phases = state.phases + self.dt * phase_deltas
        return CPGState(
            time=state.time + self.dt,
            phases=phases,
            outputs=self._outputs(phases),
        )

    def _phase_derivative(self, phases: FloatArray) -> FloatArray:
        if phases.size == 0:
            return phases
        pairwise = phases[None, :] - phases[:, None] - self.parameters.phase_biases
        weighted_sines = (
            self.parameters.coupling_weights
            * self.parameters.amplitudes[None, :]
            * np.sin(pairwise)
        )
        coupling = np.sum(weighted_sines, axis=1)
        return self.parameters.frequencies + coupling

    def _outputs(self, phases: FloatArray) -> FloatArray:
        return self.parameters.offsets + self.parameters.amplitudes * np.cos(phases)
