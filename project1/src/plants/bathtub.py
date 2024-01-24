import jax.numpy as jnp
from plants.plant import Plant
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
        """
        Return the initial state of the plant
        water_height: the initial height in the bathtub
        """
        return {"water_height": self.init_height_bathtub}

    def run_one_epoch(self, state: dict, control_signal: float, noise: float) -> (dict, float):
        """
        Update the plant's state based on the given control signal and noise 
        """
        water_height = state["water_height"]

        # The velocity of water exiting through the drain
        velocity = jnp.sqrt(2 * G * water_height)
        # Calculate the flow rate
        flow_rate = velocity * self.cross_section_drain
        # Calculate the volume change
        volume_change = control_signal + noise - flow_rate
        # Calculate the new water height in the bathtub
        water_height += volume_change / self.cross_section_area

        return {"water_height": water_height}, water_height
