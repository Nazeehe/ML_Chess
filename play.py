import chess
import torch
from state import State
from train import Net

vals = torch.load("value.pth", map_location=lambda storage, loc: storage)
model = Net()
model.load_state_dict(vals)
model.eval()

colors = { True:"White", False:"Black" }
if __name__ == "__main__":
    board = chess.Board()

    playing = True

    while playing:
        print(board)

        while True: # Loop until player makes a legal move
            print("(%s) Enter move: " % colors[board.turn])
            move = input()
            if move == "quit":
                break
            try:
                board.push_uci(move)
                break # good move
            except:
                print("That was a bad move!")

        if board.is_checkmate():
            print("[%s] played a checkmate!" % colors[not board.turn])
            playing = False
            continue

        isort = []
        for e in board.legal_moves:
            board.push(e)
            s = State(board)
            brd = s.serialize()[None]
            output = model(torch.tensor(brd).float())
            output = output.data[0][0]
            isort.append((output, e))
            board.pop()

        sortedMoves = sorted(isort, key=lambda x: x[0], reverse = board.turn)
        board.push(sortedMoves[0][1])
        if board.is_checkmate():
            print("[%s] played a checkmate!" % colors[not board.turn])
            playing = False

