from src.game_state.game_state import PublicGameState
from src.game_manager.game_stage import GameStage
from src.state_manager.manager import StateManager


class SubTree:
    def __init__(self, root: PublicGameState, end_stage: GameStage, end_depth: int):
        self.root: PublicGameState = root
        self.nodes: list(SubTree) = []
        self.end_stage: GameStage = end_stage
        self.end_depth: int = end_depth
        if end_depth > 0 and root.game_stage != end_stage:
            self.generate_subtree()

    def add_node(self, node):
        self.nodes.append(node)

    def get_node(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_root(self):
        return self.root

    def get_nodes(self):
        return self.nodes

    def generate_subtree(self):
        state_manager = StateManager(self.root)
        public_game_states = state_manager.generate_possible_states()
        for public_game_state in public_game_states:
            new_sub_state = SubTree(
                public_game_state, self.end_stage, self.end_depth - 1
            )
            self.add_node(new_sub_state)

    def __str__(self):
        return self.root

    def __repr__(self):
        return self.root


def generateInitialSubtree(state, endStage, endDepth):
    # TODO: EIRIK
    return 0
