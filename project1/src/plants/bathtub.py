from plant import Plant
import math

G = 9.81

class Bathtub(Plant):
    def __init__(self, cross_section_drain: float, cross_section_area: float, init_height_bathtub: float):
        self.cross_section_drain = cross_section_drain
        self.cross_section_area = cross_section_area
        self.init_height_bathtub = init_height_bathtub

    def reset(self) -> dict:
        return {'water_height': self.init_height_bathtub}

    def run_one_epoch(self, state: dict, control_signal: float, noise: float) -> dict:
        water_height = state['water_height']

        volume = math.sqrt(2*G*water_height)
        flow_rate = volume * self.cross_section_drain
        volume_change = control_signal + noise - flow_rate
        water_height += volume_change / self.cross_section_area

        return {'water_height': water_height}, water_height
