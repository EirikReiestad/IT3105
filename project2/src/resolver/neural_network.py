import os
import json
import numpy as np
from typing import Tuple, List, Optional
from src.game_state.game_state import PublicGameState, PublicBoardState, PublicPlayerState
from src.game_manager.game_stage import GameStage
from src.poker_oracle.deck import Card, Deck
from src.poker_oracle.oracle import Oracle
from . import resolver
from src.config import Config
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dot, Add, Concatenate, Permute

config = Config()


class NeuralNetwork:
    def __init__(self,
                 total_players: int,
                 game_stage: GameStage,
                 public_cards_size: int,
                 parent_nn: Optional['NeuralNetwork'] = None,
                 model_name=None,
                 model_path=None,
                 verbose=True):
        self.game_stage = game_stage
        self.parent_nn = parent_nn
        self.oracle = Oracle()
        self.deck = Deck(shuffle=False)

        self.model_name = model_name

        if public_cards_size == 0:
            self.random = True
        else:
            # self.random = True
            # return
            self.random = False

            if parent_nn is None:
                end_state = GameStage.Showdown
            else:
                end_state = parent_nn.game_stage

            self.networks = {
                end_state: parent_nn
            }
            self.resolver = resolver.Resolver(total_players, self.networks)

            if model_path is not None and os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
            else:
                if verbose:
                    print(f"Training model for {game_stage} stage")
                if os.path.exists(f"models/{model_name}_training_data.json"):
                    training_data = self.load_training_data(
                        f"models/{model_name}_training_data.json")
                else:
                    training_data = self.create_training_data(
                        config.data['training_cases'], total_players, public_cards_size)

                training_data["train_relative_pot"] = training_data["train_relative_pot"].reshape(-1, 1)
                    
                self.model = self.create_model(
                    len(training_data["train_p_ranges"][0]))

                self.train(
                    training_data["train_p_ranges"],
                    training_data["train_o_ranges"],
                    training_data["train_public_cards"],
                    training_data["train_relative_pot"],
                    training_data["target_value_vector_p"],
                    training_data["target_value_vector_o"],
                    config.data['epochs'],
                    config.data['batch_size'],
                )
                self.model.save(f"models/{model_name}.h5")

    def load_training_data(self, path: str):
        with open(path, 'r') as infile:
            data = infile.read()

        data = json.loads(data)

        return {
                "train_p_ranges": np.array(data["train_p_ranges"]),
                "train_o_ranges": np.array(data["train_o_ranges"]),
                "train_public_cards": np.array(data["train_public_cards"]),
                "train_relative_pot": np.array(data["train_relative_pot"]),
                "target_value_vector_p": np.array(data["target_value_vector_p"]),
                "target_value_vector_o": np.array(data["target_value_vector_o"]),
        }

    def create_training_data(self, n: int, total_players: int, public_cards_size: int):
        train_p_ranges = []
        train_o_ranges = []
        train_public_cards = []
        train_relative_pot = []
        target_value_vector_p = []
        target_value_vector_o = []
        for _ in range(n):
            deck_shuffled = Deck()
            public_cards = [deck_shuffled.pop()
                            for _ in range(public_cards_size)]
            p_range = self.create_ranges(public_cards)
            o_range = self.create_ranges(public_cards)
            # Random current bet og pot size idk om det e riktig
            pot_size = np.random.randint(10, 30)
            current_bet = np.random.randint(1, 11)
            relative_pot = np.array(pot_size / (pot_size + current_bet))

            # CHEAP METHOD
            # utility_matrix = self.oracle.utility_matrix_generator(public_cards)
            # value_vector_p = utility_matrix * p_range
            # value_vector_o = utility_matrix * o_range

            # # BOOTSTRAPPED METHOD
            public_board_state = PublicBoardState(
                cards=public_cards,
                pot=pot_size,
                highest_bet=current_bet,
                game_stage=self.game_stage
            )

            players = []

            for _ in range(total_players):
                player = PublicPlayerState(
                    np.random.randint(current_bet, 101),
                    False,
                    False,
                    np.random.randint(current_bet-1, current_bet+1)
                )
                players.append(player)

            players[np.random.choice(range(total_players))
                    ].round_bet = current_bet

            state = PublicGameState(
                player_states=players,
                board_state=public_board_state,
                game_stage=self.game_stage,
                current_player_index=np.random.choice(range(total_players)),
                buy_in=1,
                check_count=np.random.choice(range(total_players)),
                raise_count=np.random.choice(range(total_players)),
                chance_event=False
            )
            if self.parent_nn is None:
                end_game_stage = GameStage.Showdown
            else:
                end_game_stage = self.parent_nn.game_stage

            self.resolver.p_range = p_range
            self.resolver.o_range = o_range
            self.resolver.resolve(state, end_game_stage, 1, 3)

            value_vector_p = self.resolver.p_range
            value_vector_o = self.resolver.o_range

            train_p_ranges.append(p_range)
            train_o_ranges.append(o_range)
            train_public_cards.append(public_cards)
            train_relative_pot.append(relative_pot)
            target_value_vector_p.append(value_vector_p)
            target_value_vector_o.append(value_vector_o)

        train_public_cards = [self.ohe_cards(cards) for cards in train_public_cards]

        train_p_ranges = [value.tolist() for value in train_p_ranges]
        train_o_ranges = [value.tolist() for value in train_o_ranges]
        train_public_cards = [value.tolist() for value in train_public_cards]
        train_relative_pot = [value.tolist() for value in train_relative_pot]
        target_value_vector_p = [value.tolist() for value in target_value_vector_p]
        target_value_vector_o = [value.tolist() for value in target_value_vector_o]

        data = {
            "train_p_ranges": train_p_ranges,
            "train_o_ranges": train_o_ranges,
            "train_public_cards": train_public_cards,
            "train_relative_pot": train_relative_pot,
            "target_value_vector_p": target_value_vector_p,
            "target_value_vector_o": target_value_vector_o,
        }
        json_data = json.dumps(data, indent=4)

        with open(f"models/{self.model_name}_training_data.json", 'w') as outfile:
            outfile.write(json_data)

        return {
            "train_p_ranges": np.array(train_p_ranges),
            "train_o_ranges": np.array(train_o_ranges),
            "train_public_cards": np.array(train_public_cards),
            "train_relative_pot": np.array(train_relative_pot),
            "target_value_vector_p": np.array(target_value_vector_p),
            "target_value_vector_o": np.array(target_value_vector_o),
        }

    def create_ranges(self, public_cards: List[Card]):
        new_range = np.random.rand(self.oracle.get_number_of_all_hole_pairs())
        for i, pair in enumerate(self.oracle.hole_pairs):
            for card in public_cards:
                if card in pair:
                    new_range[i] = 0
        new_range /= new_range.sum()
        return new_range

    def ohe_cards(self, cards: List[Card]):
        assert type(cards) == list or type(cards) == np.ndarray, f"Cards must be a list, not {type(cards)}."
        val = np.zeros(len(self.deck.stack))
        for card in cards:
            try:
                idx = self.deck.stack.index(card)
                val[idx] = 1
            except ValueError:
                print(f"Card {card} is not in the list.")
        return val

    def create_model(self, range_size: Tuple):

        input_p_range = Input(shape=(range_size, 1))
        input_o_range = Input(shape=(range_size, 1))
        input_public_cards = Input(shape=(len(self.deck.stack), 1))
        input_pot_size = Input(shape=(1, 1))

        merged_layer = Concatenate(axis=1)(
            [input_p_range, input_o_range, input_public_cards, input_pot_size]
        )

        hidden_layer1 = Dense(64, activation="relu")(merged_layer)
        hidden_layer2 = Dense(32, activation="relu")(hidden_layer1)
        hidden_layer3 = Dense(16, activation="relu")(hidden_layer2)

        value_layer = Dense(range_size, activation='relu')(hidden_layer3)

        permuted_layer = Permute((2, 1))(value_layer)

        value_layer_p1 = Dense(1, activation='relu')(permuted_layer)
        value_layer_p2 = Dense(1, activation='relu')(permuted_layer)

        dot_product_p = Dot(axes=1)([value_layer_p1, input_p_range])
        dot_product_o = Dot(axes=1)([value_layer_p2, input_o_range])
        addition_layer = Add()([dot_product_p, dot_product_o])

        model = Model(
            inputs=[input_p_range, input_o_range,
                    input_public_cards, input_pot_size],
            outputs=[value_layer_p1, value_layer_p2, addition_layer],
        )

        model.compile(optimizer="adam", loss="mse")

        model.summary()

        return model

    def train(
        self,
        p_range_train: np.ndarray,
        o_range_train: np.ndarray,
        public_cards_train: np.ndarray,
        pot_size_train: np.ndarray,
        p_range_target: np.ndarray,
        o_range_target: np.ndarray,
        epochs: int,
        batch_size: int,
    ):
        # TODO:
        # additional_layer_target = 0 for all? where to find size tho

        history = self.model.fit(
            x=[
                p_range_train,
                o_range_train,
                public_cards_train,
                pot_size_train,
            ],
            y=[p_range_target, o_range_target, np.zeros(len(p_range_target))],
            epochs=epochs,
            batch_size=batch_size,
        )

        plt.plot(history.history["loss"])
        plt.savefig("loss.png")
        plt.show()

    def run(
        self,
        state: PublicGameState,
        stage: GameStage,
        p_range: np.ndarray,
        o_range: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        state: GameState
        stage: GameStage
        p_range: np.ndarray
            hand distribution of player
        o_range: np.ndarray
            hand distribution of opponent

        Returns
        -------
        np.ndarray: The expected value of the game for the player
        """
        if np.isnan(np.min(p_range)):
            raise ValueError("Player hand distribution is NaN")
        if np.isnan(np.min(o_range)):
            raise ValueError("Opponent hand distribution is NaN")

        # RANDOM VERSION
        if self.random:
            p_random = np.random.rand(*p_range.shape)
            o_random = np.random.rand(*o_range.shape)
            return p_random, o_random
        else:
            # Cheap/Hard method
            public_cards_ohe = self.ohe_cards(state.board_state.cards)
            predicted_p_values, predicted_o_values, predicted_addition_layer = (
                self.model(
                    [
                        np.array([p_range]),
                        np.array([o_range]),
                        np.array([public_cards_ohe]),
                        np.array([np.array(
                            [state.board_state.pot/(state.board_state.pot + state.board_state.highest_bet)])]),
                    ],
                )
            )

            predicted_p_values = predicted_p_values[0]
            predicted_o_values = predicted_o_values[0]
            # if predicted_addition_layer != 0:
            #     print("Addition", predicted_addition_layer)
            #     print("P values", predicted_p_values)
            #     print("O values", predicted_o_values)
            #     raise ValueError("Result of addition layer should be 0")

            return predicted_p_values, predicted_o_values
