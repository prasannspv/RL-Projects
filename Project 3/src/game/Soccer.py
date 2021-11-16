import numpy as np
from copy import deepcopy


class Soccer:
    def __init__(self, player_a_state, player_b_state, seed=7):
        self.init_state = (player_a_state, player_b_state)
        self.players = {
            "A": SoccerPlayer(player_a_state, False),
            "B": SoccerPlayer(player_b_state, True)
        }
        self.actions = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1),   # RIGHT
            4: (0, 0)    # STICK
        }
        np.random.seed(seed)

    def sample(self):
        return np.random.choice(list(self.actions.keys()))

    def act(self, actions):
        action_a, action_b = actions
        new_a_pos = self.transition("A", action_a)
        new_b_pos = self.transition("B", action_b)

        if np.random.uniform(0, 1) <= 0.5:
            # 0.5 prob A will go first and then B
            self.players = self.get_next_state("A", new_a_pos, new_b_pos)
        else:
            # 0.5 prob B will go first and then A
            self.players = self.get_next_state("B", new_a_pos, new_b_pos)

    def transition(self, player, action):
        x, y = self.players[player].position
        action_delta = self.actions[action]
        new_x = np.clip(x + action_delta[0], 0, 1)
        new_y = np.clip(y + action_delta[1], 0, 3)
        return new_x, new_y

    def get_next_state(self, player, new_A_pos, new_B_pos):
        if player == "A":
            player_first = deepcopy(self.players["A"])
            player_next = deepcopy(self.players["B"])
            new_first_pos = new_A_pos
            new_next_pos = new_B_pos
            other_player = "B"
        else:
            player_first = deepcopy(self.players["B"])
            player_next = deepcopy(self.players["A"])
            new_first_pos = new_B_pos
            new_next_pos = new_A_pos
            other_player = "A"

        if new_first_pos == player_next.position:
            if player_first.kicks:
                SoccerPlayer.handle_collision(player_first, player_next)
        elif new_first_pos == new_next_pos:
            player_first.position = new_first_pos
            if player_next.kicks:
                SoccerPlayer.handle_collision(player_first, player_next)
        else:
            player_first.position = new_first_pos
            player_next.position = new_next_pos

        return {
            player: player_first,
            other_player: player_next
        }

    def get_reward(self, player):
        x, y = self.players[player].position
        if y == 0:
            if player == "A":
                return 100
            else:
                return -100
        elif y == 3:
            if player == "A":
                return -100
            else:
                return 100
        else:
            return 0

    def reset(self):
        self.players = {
            "A": SoccerPlayer(self.init_state[0], False),
            "B": SoccerPlayer(self.init_state[1], True)
        }

    def encode_state(self, player):
        return {
            (0, 0): "T1",
            (0, 1): "T2",
            (0, 2): "T3",
            (0, 3): "T4",
            (1, 0): "B1",
            (1, 1): "B2",
            (1, 2): "B3",
            (1, 3): "B4"
        }[self.players[player].position]

    # def get_next_possible_states(self, player):
    #     return [self.transition(player, action) for action in self.actions.keys()]


class SoccerPlayer:
    def __init__(self, position, kicks):
        self.position = position
        self.kicks = kicks

    @staticmethod
    def handle_collision(player_a, player_b):
        player_a.kicks, player_b.kicks = player_b.kicks, player_a.kicks
