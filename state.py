import chess
import chess.pgn
import numpy as np


class State(object):
    def __init__(self, board=None) -> None:
        super().__init__()
        self.board = board

    def serialize(self):
        rep = np.zeros((13,8,8), np.uint8)
        for pos in range(64):
            pp = self.board.piece_at(pos)
            if pp is None:
                continue
            index = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5, "p": 6, "n":7, "b":8, "r":9, "q":10, "k": 11}[pp.symbol()]
            x = pos // 8
            y = pos % 8
            rep[index][x][y] = 1
        rep[12] = self.board.turn * 1.0
        return rep


