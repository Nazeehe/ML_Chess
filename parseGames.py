import chess
import chess.pgn
import numpy as np
from state import State

num_samples = 500000

def GenerateData(inFile, outFile, numSamples):
    print("Generating %s with %d samples..." % (outFile, numSamples))
    X, Y = [], []
    values = {'1/2-1/2':0, '0-1':-1, '1-0':1}
    gn = 0
    pgn = open(inFile)
    while 1:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        res = game.headers["Result"]
        if res not in values:
            continue

        value = values[res]
        board = game.board()

        for i, move in enumerate(game.mainline_moves()):
            board.push(move)
            s = State(board)
            rep = s.serialize()
            X.append(rep)
            Y.append(value)

        gn += 1
        if len(X) >= numSamples:
            break

    X = np.array(X)
    Y = np.array(Y)
    np.savez(outFile, X, Y)

GenerateData("D:/Code/mlearn/chess/pgn/KingBase2019-A00-A39.pgn", "data_10k.npz", 10000)
GenerateData("D:/Code/mlearn/chess/pgn/KingBase2019-E60-E99.pgn", "TestSet.npz", 15000)
