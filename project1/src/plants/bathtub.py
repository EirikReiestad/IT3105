import jax.numpy as jnp
from plants.plant import Plant
from lib import jax_type_to_python_type
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


G = 9.81


class Bathtub(Plant):
    def __init__(
        self,
        cross_section_area: float,
        cross_section_drain: float,
        init_height_bathtub: float,
    ):
        self.cross_section_drain = cross_section_drain
        self.cross_section_area = cross_section_area
        self.init_height_bathtub = init_height_bathtub

    def reset(self) -> dict:
        return {"water_height": self.init_height_bathtub}

    def run_one_epoch(self, state: dict, control_signal: float, noise: float) -> (dict, float):
        water_height = state["water_height"]

        volume = jnp.sqrt(2 * G * water_height)
        flow_rate = volume * self.cross_section_drain
        volume_change = control_signal + noise - flow_rate
        water_height += volume_change / self.cross_section_area

        water_height = jax_type_to_python_type(water_height)

        return {"water_height": water_height}, water_height
