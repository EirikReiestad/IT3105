from src.game_state.game_state import PublicGameState
from src.game_manager.game_stage import GameStage
from src.state_manager.manager import StateManager
from src.poker_oracle.oracle import Oracle
import numpy as np


class Node:
    depth: int

    def __init__(self,
                 state: PublicGameState,
                 end_stage: GameStage,
                 end_depth: int,
                 depth: int):
        self.state: PublicGameState = state
        self.state_manager = StateManager(self.state)

        num_all_hole_pairs = Oracle.get_number_of_all_hole_pairs()
        self.strategy: np.ndarray = np.ones(
            (num_all_hole_pairs, self.state_manager.get_num_legal_actions()))

        # sigma_s = np.zeros((num_all_hole_pairs,
        # state_manager.get_num_legal_actions()))
        #
        self.children: list(Node) = []
        self.end_stage: GameStage = end_stage
        self.end_depth: int = end_depth
        self.depth = depth
        if depth < end_depth and state.game_stage != end_stage:
            self.generate_child_node()

    def add_child(self, node):
        self.children.append(node)

    def get_node(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_root(self):
        return self.root

    def get_nodes(self):
        return self.nodes

    def generate_child_node(self):
        public_game_states = self.state_manager.generate_possible_states()
        for public_game_state in public_game_states:
            new_sub_state = Node(
                public_game_state, self.end_stage, self.end_depth - 1, self.depth + 1
            )
            self.add_child(new_sub_state)

    def __str__(self):
        return self.root

    def __repr__(self):
        return self.root
