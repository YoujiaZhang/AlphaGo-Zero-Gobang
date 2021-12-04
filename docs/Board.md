# 棋盘 + 游戏
这里主要是我们如何 **描述** 一个棋盘，以及一场游戏。
## 1 描述棋盘
### 1.1 主要变量
```
Board.py

width：棋盘宽
height：棋盘高
states：一场对局各方落子的记录    
——————————————————————————————————————————
棋盘状态《states》存储在dict中
key: 棋盘上的落子行为 
value: 玩家
{38: 1, 23: 2, 24: 1, 36: 2, ··· }
玩家1 落子 38 位置
玩家2 落子 23 位置 ···
——————————————————————————————————————————
n_in_row：n子棋
players：玩家列表 [1, 2] # player1 和 player2
current_player：当前玩家
availables：棋盘剩余可落子的空位
last_move：最新的一次落子
```
### 1.2 主要函数
```
Board.py

def do_move(self, move):
    """
    落子
    move：落子的位置 一个整数取值范围 0 到 H*W-1
    """
    self.states[move] = self.current_player
    self.availables.remove(move)  # 减少一个可以落子的位置

    # 切换玩家角色
    self.current_player = (
        self.players[0] if self.current_player == self.players[1]
        else self.players[1]
    )
    self.last_move = move # 记录上一次落子的位置


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
```
---
## 2 描述游戏
### 2.1 主要函数
```
Game.py

def selfPlay(self, player, Index=0):
    """ 
    构建一个AI自我博弈，重用搜索树，并存储自玩数据：(棋盘状态, 落子概率, 胜者预测) 以供训练
    """
    
    self.board.initBoard() # 初始化一个棋盘
    boards, probs, currentPlayer = [], [], [] # 用以存储相关对局信息

    while True:
        # 当前玩家，针对当前棋盘根据MCTS获取下一步的落子行为
        move, move_probs = player.getAction(self.board, self.flag_is_train)
        
        # 存储下棋的数据
        boards.append(self.board.current_state())
        probs.append(move_probs)
        currentPlayer.append(self.board.current_player)

        # 下棋落子
        self.board.do_move(move)

        # # 展示下棋的过程
        if self.flag_is_shown:
            self.Show(self.board)
        
        # 是否已经结束
        gameOver, winner = self.board.gameIsOver()

        if gameOver:
            # 根据最终游戏的结果，构造用于训练神经网络的《标签》Z
            winners_z = np.zeros(len(currentPlayer))
            if winner != -1:
                winners_z[np.array(currentPlayer) == winner] = 1.0
                winners_z[np.array(currentPlayer) != winner] = -1.0

            # 重新设置MCTS，初始化了整棵树
            player.resetMCTS()

            # 在GUI上显示一些相关信息而已
            if self.flag_is_shown:
                if winner != -1:
                    if self.flag_is_train == False:
                        playerName = 'you'
                        if self.board.current_player != 1:
                            playerName = 'AI'
                    else:
                        playerName = 'AI-'+str(self.board.current_player)
                    self.drawText("Game end. Winner is :"+str(playerName))
                else:
                    self.drawText("Game end. Tie")

            # 这个 rect 用来圈（红圈）出最新一次落子的位置
            self.rect = None

            # 返回的这些数据都是很有用的 是神经网络的《学习资料》
            return winner, zip(boards, probs, winners_z)
```


