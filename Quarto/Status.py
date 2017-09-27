import numpy as np
import matplotlib.pyplot as plt
from .Gameboard import *
import random
import time

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

letters = list(items.keys())

# def left_items_image(left_items):
#     im = np.zeros((4,4,4))
#     for it in left_items:
#         pos = items_image[it]
#         im[pos[0],pos[1]] = items[it]
#     return im

def get_initial_status():
    letter = random.choice(letters)
    left_letters = letters[:]
    left_letters.remove(letter)
    return Status(Gameboard(),letter, left_letters)


def show_action_image(image, save=False):
    f,(a1,a2,a3) = plt.subplots(4,1)
    a1.imshow(image[:,:,0], cmap="gray",vmin=0, vmax=255)
    a2.imshow(image[:,:,1], cmap="gray",vmin=0, vmax=255)
    a3.imshow(image[:,:,2], cmap="gray",vmin=0, vmax=255)
    if save:
        plt.savefig("actions/action{}.png".format(time.time()))
    else:
        plt.show()



class Status :

    # status is (gameboard, item, left_items, image_left_items).
    def __init__(self, gameboard, item, left_items):
        self.gameboard = gameboard
        self.item = item
        self.left_items = left_items
        #self.left_items_image = left_image


    # action is ((move), element) -> ( (coord, element), element) -> ( ((0,1), "A"), "B" )
    def get_available_actions(self):
        results = []
        for pos in self.gameboard.spaces:
            for next_item in self.left_items:
                results.append(((pos,self.item), next_item))
        return np.array(results)

    def get_new_status(self, action):
        if (self.item != action[0][1]):
            print("Appling NON consecutive actions!")
            return None
        new_left_items = self.left_items[:]
        new_left_items.remove(action[1])
        #the element of the action is already out of the leftimage
        #new_left_image = self._remove_item_leftimage(action[1])
        new_status = Status(self.gameboard.apply_move_copy(action[0]),
                            action[1], new_left_items)
        #Check if we have win
        win = new_status.gameboard.check_win()
        win_reward = self.gameboard.reward(action[0]) if win else 0
        return (new_status, win, win_reward)

    def get_transition_reward(self, action1, action2):
        #applying second move on top of the first one
        reward1, second_board = self.gameboard.reward( action1[0], newboard=True)
        reward2 = second_board.reward(action2[0])
        return reward1 - reward2


    # def _remove_item_leftimage(self, item):
    #     pos = items_image[item]
    #     im = np.copy(self.left_items_image)
    #     im[pos[0],pos[1]] = [0,0,0,0]
    #     return im

    def get_action_image(self, action):
        total = np.empty((4,16,3))
        total[:,:,0] = np.concatenate(self.gameboard.gb[:,:].T, axis=1)
        #we don't check consecutive actions for performance reasons
        gb = self.gameboard.apply_move_copy(action[0])
        total[:,:,1] = np.concatenate(gb.gb[:,:].T, axis=1)
        #on copy of spaces because we are chaging it
        for sp in gb.spaces:
            gb.apply_move((sp, action[1]),nospaces=True)

        total[:,:,2] = np.concatenate(gb.gb[:,:].T,     axis =1)
        total[total==1] = 255
        total[total==-1] = 125
        return total

    def get_action_images(self, actions):
        result = []
        for act in actions:
            result.append(self.get_action_image( act))
        return np.array(result)

    def show(self):
        print("Gameboard")
        self.gameboard.show()
        print("Spaces: ",self.gameboard.spaces)
        print("Item {}, LeftItems: {}".format(self.item, "".join(self.left_items)))

    def get_num_used_pieces(self):
        return 16 - len(self.left_items)
