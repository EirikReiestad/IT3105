import numpy as np
from typing import Tuple, List
from src.game_state.game_state import PublicGameState
from src.game_manager.game_stage import GameStage
from src.poker_oracle.deck import Card, Deck
from src.poker_oracle.oracle import Oracle

from keras.models import Model
from keras.layers import Input, Dense, Dot, Add, Concatenate, Reshape, Permute


# TODO!! EIRIKKKKKKKKKKKKKKKKOSELIG
class NeuralNetwork:
    # TODO: each stage has a neural network since different public card sizes
    def __init__(self, public_cards_size: int):
        self.count = 0
        if public_cards_size == 0:
            self.random = True
        else:
            self.random = False
            # CONFIG? here 10000 training cases
            training_data = self.create_training_data(2, public_cards_size)
            self.model = self.create_model(len(training_data["train_p_ranges"][0]))
            training_data["train_relative_pot"] = training_data[
                "train_relative_pot"
            ].reshape(-1, 1)

            self.train(
                training_data["train_p_ranges"],
                training_data["train_o_ranges"],
                np.array(
                    [
                        NeuralNetwork.ohe_cards(public_cards)
                        for public_cards in training_data["train_public_cards"]
                    ]
                ),
                training_data["train_relative_pot"],
                training_data["target_value_vector_p"],
                training_data["target_value_vector_o"],
                ## TODO: add in config?
                10,
                32,
            )

    def create_training_data(self, n: int, public_cards_size: int):
        hole_pairs = Oracle.generate_all_hole_pairs()
        train_p_ranges = []
        train_o_ranges = []
        train_public_cards = []
        train_relative_pot = []
        target_value_vector_p = []
        target_value_vector_o = []
        for _ in range(n):
            deck_shuffled = Deck()
            public_cards = [deck_shuffled.pop() for _ in range(public_cards_size)]
            p_range = NeuralNetwork.create_ranges(hole_pairs, public_cards)
            o_range = NeuralNetwork.create_ranges(hole_pairs, public_cards)
            utility_matrix = Oracle.utility_matrix_generator(public_cards)
            value_vector_p = utility_matrix * p_range
            value_vector_o = utility_matrix * o_range
            # Random current bet og pot size idk om det e riktig
            pot_size = np.random.randint(10, 101)
            current_bet = np.random.randint(1, 11)
            relative_pot = np.array(pot_size / (pot_size + current_bet))

            train_p_ranges.append(p_range)
            train_o_ranges.append(o_range)
            train_public_cards.append(public_cards)
            train_relative_pot.append(relative_pot)
            target_value_vector_p.append(value_vector_p)
            target_value_vector_o.append(value_vector_o)

        return {
            "train_p_ranges": np.array(train_p_ranges),
            "train_o_ranges": np.array(train_o_ranges),
            "train_public_cards": np.array(train_public_cards),
            "train_relative_pot": np.array(train_relative_pot),
            "target_value_vector_p": np.array(target_value_vector_p),
            "target_value_vector_o": np.array(target_value_vector_o),
        }

    @staticmethod
    def create_ranges(all_pairs: List[Card], public_cards: List[Card]):
        new_range = np.random.rand(len(all_pairs))
        new_range /= new_range.sum()
        for i, pair in enumerate(all_pairs):
            for card in public_cards:
                if card in pair:
                    new_range[i] = 0
        return new_range

    @staticmethod
    def ohe_cards(cards: List[Card]):
        deck = Deck(shuffle=False)
        val = np.zeros(len(deck.stack))
        for card in cards:
            try:
                idx = deck.stack.index(card)
                val[idx] = 1
            except ValueError:
                print(f"Card {card} is not in the list.")
        return val

    @staticmethod
    def create_model(range_size: Tuple):

        deck = Deck(shuffle=False)
        input_p_range = Input(shape=(range_size, 1))
        input_o_range = Input(shape=(range_size, 1))
        input_public_cards = Input(shape=(len(deck.stack), 1))
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
            inputs=[input_p_range, input_o_range, input_public_cards, input_pot_size],
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

        self.model.fit(
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

        ### RANDOM VERSION
        if self.random:
            p_random = np.random.rand(*p_range.shape)
            o_random = np.random.rand(*o_range.shape)
            return p_random, o_random
        else:
            #### Cheap method
            public_cards_ohe = NeuralNetwork.ohe_cards(state.board_state.cards)
            predicted_p_values, predicted_o_values, predicted_addition_layer = (
                self.model.predict(
                    [
                        np.array([p_range]),
                        np.array([o_range]),
                        np.array([public_cards_ohe]),
                        np.array([np.array([state.board_state.pot])]),
                    ],
                    verbose=False
                )
            )

            predicted_p_values = predicted_p_values[0]
            predicted_o_values = predicted_o_values[0]
            self.count += 1
            print(self.count)
            # if predicted_addition_layer != 0:
            #     print("Addition", predicted_addition_layer)
            #     print("P values", predicted_p_values)
            #     print("O values", predicted_o_values)
            #     raise ValueError("Result of addition layer should be 0")

            return predicted_p_values, predicted_o_values
