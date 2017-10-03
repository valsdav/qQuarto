import numpy as np
import matplotlib.pyplot as plt

items = {
    "A": (1,1,1,1),
    "B": (1,1,1,-1),
    "C": (1,1,-1,1),
    "D": (1,1,-1,-1),
    "E": (1,-1,1,1),
    "F": (1,-1,1,-1),
    "G": (1,-1,-1,1),
    "H": (1,-1,-1,-1),
    "I": (-1,1,1,1),
    "J": (-1,1,1,-1),
    "K": (-1,1,-1,1),
    "L": (-1,1,-1,-1),
    "M": (-1,-1,1,1),
    "N": (-1,-1,1,-1),
    "O": (-1,-1,-1,1),
    "P": (-1,-1,-1,-1)
}

items_image = {
    "A": (0,0),
    "B": (0,1),
    "C": (0,2),
    "D": (0,3),
    "E": (1,0),
    "F": (1,1),
    "G": (1,2),
    "H": (1,3),
    "I": (2,0),
    "J": (2,1),
    "K": (2,2),
    "L": (2,3),
    "M": (3,0),
    "N": (3,1),
    "O": (3,2),
    "P": (3,3)
}

class Gameboard:

    def __init__(self, gameboard = None):
        if(gameboard != None):
            self.gb = np.copy(gameboard.gb)
            self.spaces = gameboard.spaces[:]
        else:
            self.gb = np.zeros((4,4,4))
            self.spaces = [(0,0),(0,1),(0,2),(0,3),
                           (1,0),(1,1),(1,2),(1,3),
                           (2,0),(2,1),(2,2),(2,3),
                           (3,0),(3,1),(3,2),(3,3)]

    def score(self):
        sum = 0
        for i in range(4):
            sum_plane = 0
            for j in range(4):
                sum_row = 0
                for k in range(4):
                    sum_row += self.gb[j,k,i]
                if(abs(sum_row) == 4):
                    sum_plane += 3
                elif(abs(sum_row) ==3):
                    sum_plane += 1
                elif(abs(sum_row) ==2):
                    sum_plane += 0.5
                elif(abs(sum_row) ==1):
                    sum_plane += 0.1
            sum+=sum_plane
        #now diagonals
        for i in range(4):
            sum_diaga = 0
            sum_diagb = 0
            for k in range(4):
                sum_diaga += self.gb[j,k,i]
                sum_diagb += self.gb[j,k,i]
            if(abs(sum_diaga) == 4):
                sum += 3
            elif(abs(sum_diaga) ==3):
                sum += 1
            elif(abs(sum_diaga) ==2):
                sum += 0.5
            elif(abs(sum_diaga) ==1):
                sum += 0.1
            #second diagonal
            if(abs(sum_diagb) == 4):
                sum += 3
            elif(abs(sum_diagb) ==3):
                sum += 1
            elif(abs(sum_diagb) ==2):
                sum += 0.5
            elif(abs(sum_diagb) ==1):
                sum += 0.1
        return sum

    def check_win(self):
        win = False
        for i in range(4):
            for j in range(4):
                sum_row = 0
                for k in range(4):
                    sum_row += self.gb[j,k,i]
                if(abs(sum_row) == 4):
                    win = True
        #now diagonals
        for i in range(4):
            sum_diaga = 0
            sum_diagb = 0
            for k in range(4):
                sum_diaga += self.gb[k,k,i]
                sum_diagb += self.gb[k,3-k,i]
            if(abs(sum_diaga) == 4):
                win = True
            #second diagonal
            if(abs(sum_diagb) == 4):
                win = True
        return win

    def apply_move_copy(self, move):
        gameb = Gameboard(self)
        gameb.apply_move(move)
        return gameb

    def apply_moves_copy(gameboard, moves):
        gameb = Gameboard(self)
        gameb.apply_moves(moves)
        return gameb

    def apply_move(self, move, nospaces=False):
        self.gb[move[0][0], move[0][1]] = items[move[1]]
        if not nospaces:
            self.spaces.remove((move[0][0], move[0][1]))

    def apply_moves(self, moves):
        for move in moves:
            self.apply_move(move)

    def reward(self, move, newboard=False):
        initial_score = self.score()
        second_board = self.apply_move_copy(move)
        second_score = second_board.score()
        if newboard:
            return (second_score - initial_score), second_board
        else:
            return (second_score - initial_score)

    def show(self):
        gg = Gameboard(self)
        gg.gb[gg.gb==1] = 255
        gg.gb[gg.gb==-1] = 125
        f,(a1,a2,a3,a4) = plt.subplots(1,4)
        a1.imshow(gg.gb[:,:,0], cmap="gray",vmin=0, vmax=255)
        a2.imshow(gg.gb[:,:,1], cmap="gray",vmin=0, vmax=255)
        a3.imshow(gg.gb[:,:,2], cmap="gray",vmin=0, vmax=255)
        a4.imshow(gg.gb[:,:,3], cmap="gray",vmin=0, vmax=255)
        plt.show()

    def fullcells(self):
        cells = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                if not np.array_equal(self.gb[i,j],[0,0,0,0]):
                    cells[i,j] = 1
        return cells

    def voidcells(self):
        cells = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                if np.array_equal(self.gb[i,j],[0,0,0,0]):
                    cells[i,j] = 1
        return cells

    def show_full_cells(self):
        t = self.fullcells()
        t[t==1] = 255
        plt.imshow(t, cmap="gray", vmax=255)
        plt.show()

    def show_void_cells(self):
        t = self.voidcells()
        t[t==1] = 255
        plt.imshow(t, cmap="gray", vmax=255)
        plt.show()

    def __str__(self):
        lines = []
        for i in range(4):
            cols = []
            for j in range(4):
                item = "0"
                for move , arr in items.items():
                    if np.all(self.gb[i,j] == arr):
                        item = move
                cols.append(item)
            lines.append(" ".join(cols))
        return "\n".join(lines)

    def print_num(self):
        lines = []
        for i in range(4):
            cols = []
            for j in range(4):
                item = ""
                for k in range(4):
                    item += str(int(self.gb[i,j,k]))
                cols.append(item)
            lines.append(" ".join(cols))
        return "\n".join(lines)
