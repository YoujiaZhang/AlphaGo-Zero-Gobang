from __future__ import print_function
import numpy as np

class Board(object):
    """
    棋盘
    """
    def __init__(self, width=8, height=8, n_in_row=5):
        # 设置棋盘的长宽
        self.width = width
        self.height = height

        # 棋盘状态《states》存储在dict中
        # key: 棋盘上的落子行为 value: 玩家
        # {38: 1, 23: 2, 24: 1, 36: 2, ··· }
        # 玩家1 落子 38 位置
        # 玩家2 落子 23 位置 ···
        self.states = {}
        
        self.n_in_row = n_in_row # 设置多少棋子连在一起获胜 (默认五子棋)
        self.players = [1, 2]    # player1 和 player2

    def initBoard(self, start_player=0):
        """
        初始化棋盘
        start_player: 先手玩家
        """
        # 合理性判断
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('板宽和板高不得小于{}'.format(self.n_in_row))
        
        # 当前玩家设置
        #   默认先手的是玩家1
        self.current_player = self.players[start_player]

        # 在列表中保留还没有落子的位置
        self.availables = list(range(self.width * self.height))
        self.states = {}

        # 刚刚落子的情况
        self.last_move = -1

    def current_state(self):
        """
        以当前 玩家player 角度返回当前棋盘
        """
        # 使用 4*W*H 存储棋盘的状态
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            # moves 数组
            # 记录着两个玩家交错的落子位置

            move_curr = moves[players == self.current_player] # 当前玩家落子的位置
            move_oppo = moves[players != self.current_player] # 对手玩家落子的位置

            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0 # 当前玩家所有落子棋盘
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0 # 对手玩家所有落子棋盘

            # 标记最近一次落子位置
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0

        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0 

        return square_state[:, ::-1, :]

    def do_move(self, move):
        """
        落子
        move：落子的位置 一个整数 0 ~ H*W-1
        """
        self.states[move] = self.current_player
        self.availables.remove(move)  # 减少一个可以落子的位置

        # 切换玩家角色
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move # 记录上一次落子的位置

    def has_a_winner(self):
        """
        判断是否比出输赢
        """
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def gameIsOver(self):
        """
        判断游戏是否结束
        """
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def getCurrentPlayer(self):
        return self.current_player