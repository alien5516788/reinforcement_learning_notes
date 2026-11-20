from dataclasses import dataclass
from enum import Enum
from itertools import product

WIDTH = 5
HEIGHT = 5


# State (S)
#
@dataclass(frozen=True)
class State:
    robot: tuple[int, int]
    dock: tuple[int, int]

    @staticmethod
    def all() -> list["State"]:
        positions = list(product(range(WIDTH), range(HEIGHT)))

        return [State(robot, dock) for robot in positions for dock in positions]

    def is_terminal(self) -> bool:
        return self.robot == self.dock


# Action (A)
#
class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @staticmethod
    def all() -> list["Action"]:
        return list(Action)


# World model (P)
#
#   P(s' | s, a)
#
def world_model(s: State, a: Action, s_next: State) -> float:
    if s.dock != s_next.dock:
        return 0.0

    # Terminal state
    if s.is_terminal():
        return 1.0 if s == s_next else 0.0

    x_diff = s.robot[0] - s_next.robot[0]
    y_diff = s.robot[1] - s_next.robot[1]

    match a:
        case Action.UP:
            if x_diff != 0:
                return 0.0
            elif y_diff == 1 or (y_diff == 0 and s.robot[1] == 0):
                return 1.0
            else:
                return 0.0

        case Action.DOWN:
            if x_diff != 0:
                return 0.0
            elif y_diff == -1 or (y_diff == 0 and s.robot[1] == HEIGHT - 1):
                return 1.0
            else:
                return 0.0

        case Action.LEFT:
            if y_diff != 0:
                return 0.0
            elif x_diff == 1 or (x_diff == 0 and s.robot[0] == 0):
                return 1.0
            else:
                return 0.0

        case Action.RIGHT:
            if y_diff != 0:
                return 0.0
            elif x_diff == -1 or (x_diff == 0 and s.robot[0] == WIDTH - 1):
                return 1.0
            else:
                return 0.0

    return 0.0


# Reward (R)
#
#   R(s, a, s')
#
def reward(s: State, a: Action, s_next: State) -> float:
    # Terminal state
    if s.is_terminal():
        return 0.0

    # Reached the dock
    if s_next.is_terminal():
        return 10.0

    # Hit a wall
    if s == s_next:
        return -2.0

    # Normal movement
    return -1.0


# Gamma (γ)
# Discount factor
#
GAMMA: float = 0.9


def print_v_table(v_table: dict[State, float], dock: tuple[int, int]):
    print(f"Dock: x = {dock[0]} y = {dock[1]}")

    for y in range(HEIGHT):
        for x in range(WIDTH):
            state = State(robot=(x, y), dock=dock)

            value = v_table.get(state, 0.0)

            print(f"{value:7.2f}", end="")

        print()
