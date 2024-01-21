from plants.plant import Plant
import jax
import jax.numpy as jnp


class Cournot(Plant):
    def __init__(self, max_price, marginal_cost):
        self.max_price = max_price
        self.marginal_cost = marginal_cost

    def reset(self):
        """
        Return the initial state of the plant
        P: amount of product 1
        q2: amount of product 2
        """
        return {"q1": 0, "q2": 0}

    def run_one_epoch(self, state: dict, control_signal: float, noise: float) -> (dict, float):
        """
        Update the plant's state based on the given control signal and noise
        """
        # jax.debug.print("control signal: {cs}", cs=control_signal)
        state = state.copy()
        # 1. q1 updates based on U.
        state["q1"] = control_signal + state["q1"]
        state["q1"] = jnp.clip(state["q1"], 0, 1)
        # 2. q2 updates based on D.
        state["q2"] = noise + state["q2"]
        state["q2"] = jnp.clip(state["q2"], 0, 1)
        # 3. q = q1 + q2
        q = state["q1"] + state["q2"]
        # 4. p(q) = pmax − q
        price = self.max_price - q
        profit = state["q1"] * (price - self.marginal_cost)

        return state, profit
