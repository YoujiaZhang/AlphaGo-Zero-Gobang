import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
from Board import *

class Game():
    """
    管理整个游戏
    """
    game_name = "Gobang"      # 游戏名
    board_width =  8          # 棋盘宽度
    board_height = 8          # 棋盘高度
    n_in_row = 5              # N子棋
    flag_human_click = False  # 
    move_human = -1           #
    piece_size = 0.4          #
    margin = 0.5              #

    def __init__(self, flag_is_shown = True, flag_is_train = True):
        self.flag_is_shown = flag_is_shown
        self.flag_is_train = flag_is_train

        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        fig = plt.figure(self.game_name)
        plt.ion()   #enable interaction
        self.ax = plt.subplot(111)
        self.canvas = fig.canvas
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.set_bg()

    def onclick(self, event):
        x, y = event.xdata, event.ydata
        try:
            x = int(np.round(x))
            y = int(np.round(y))
        except TypeError:
            print("Invalid area")
        else:
            self.move_human = x*self.board_width + y
            self.flag_human_click = True

    def set_bg(self):
        plt.cla()
        plt.title(self.game_name)
        plt.grid(linestyle='-')
        plt.axis([-self.margin, self.board.width - self.margin, - self.margin, self.board.height - self.margin])
        x_major_locator=MultipleLocator(1)
        y_major_locator=MultipleLocator(1)
        self.ax.xaxis.set_major_locator(x_major_locator)
        self.ax.yaxis.set_major_locator(y_major_locator)

    def graphic(self, board):
        """
        绘制棋盘并显示游戏信息
        """
        if board.current_player == 1:
            color='blue'
        elif board.current_player == 2:
            color='black'
        x = board.last_move // board.width
        y = board.last_move % board.height
        plt.text(x, y, ("%d" % len(board.states)), c = "white")
        self.ax.add_artist(plt.Circle((x, y), self.piece_size, color= color))
        plt.pause(0.001)

    def start_self_play(self, player):
        """ 
        构建一个AI自我博弈，重用搜索树，并存储自玩数据：(state，MCTS_probs，z) 以供训练
        """
        self.board.init_board()
        states, mcts_probs, current_players = [], [], []

        while True:
            move, move_probs = player.get_action(self.board, self.flag_is_train)
            
            # 存储下棋的数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)

            # 下棋
            self.board.do_move(move)

            # 展示下棋的过程
            if self.flag_is_shown:
                self.graphic(self.board)
            
            # 是否已经结束
            end, winner = self.board.game_end()

            if end:
                # 从每个《状态》当前玩家的角度看赢家
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0

                # 更新设置MCTS
                player.reset_player()

                if self.flag_is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")

                self.set_bg()
                return winner, zip(states, mcts_probs, winners_z)

    def start_play(self, player):
        """ 
        与人对弈
        """
        # 先手玩家为 第0号玩家
        self.board.init_board(0)

        while True:
            if self.board.current_player == 1:
                move, move_probs = player.get_action(self.board, self.flag_is_train)
                # 落子
                self.board.do_move(move)
            else:
                if self.flag_human_click:
                    if self.move_human in self.board.availables:
                        self.flag_human_click = False
                        self.board.do_move(self.move_human)
                    else:
                        self.flag_human_click = False
                        print("Invalid input")

            if self.flag_is_shown:
                self.graphic(self.board)

            end, winner = self.board.game_end()

            if end:

                # 更新设置MCTS
                player.reset_player()
                if self.flag_is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                self.set_bg()
                break