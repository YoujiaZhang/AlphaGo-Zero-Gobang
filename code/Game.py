import numpy as np
from Board import *
from tkinter import END
from PIL import Image, ImageGrab

class Game():
    """
    管理整个游戏
    """
    boardWidth =  8           # 棋盘宽度
    boardHeight = 8           # 棋盘高度
    n_in_row = 5              # N子棋
    flag_human_click = False  # 
    move_human = -1           #

    def __init__(self, Canvas, scrollText, flag_is_shown = True, flag_is_train = True):
        self.flag_is_shown = flag_is_shown
        self.flag_is_train = flag_is_train

        self.board = Board(width=self.boardWidth, height=self.boardHeight, n_in_row=self.n_in_row)

        self.Canvas = Canvas
        self.scrollText = scrollText
        self.rect = None


    def Show(self, board, KEY=False):
        """
        绘制棋盘并显示游戏信息
        """
        x = board.last_move // board.width
        y = board.last_move % board.height
        self.drawPieces(player=board.current_player, rc_pos=(x,y), Index=len(board.states))
        
        if KEY:
            if self.flag_is_train == False:
                playerName = 'you'
                if board.current_player != 1:
                    playerName = 'AI'
            else:
                playerName = 'AI-'+str(board.current_player)

            self.drawText(str(len(board.states))+' '+playerName+':'+str(x)+' '+str(y))


    def drawText(self, string):
        self.scrollText.insert(END, string+'\n')
        self.scrollText.see(END)
        self.scrollText.update()


    def drawPieces(self, player, rc_pos, Index, RADIUS=15, draw_rect=True):
        x, y = self.convert_rc_to_xy(rc_pos)
        colorText  = 'black' if player == 1 else 'white'
        colorPiece = 'white' if player == 1 else 'black'
        self.Canvas.create_oval(x-RADIUS, y-RADIUS, x + RADIUS, y+RADIUS, fill=colorPiece, outline=colorPiece)
        if draw_rect == True:
            if self.rect == None:
                OFFSET = 20
                self.rect = self.Canvas.create_rectangle(x-OFFSET, y-OFFSET, x+OFFSET, y+OFFSET, outline="#c1005d")
                self.rect_xy_pos = (x, y)
            else:
                rc_pos = self.convert_xy_to_rc((x, y))
                old_x, old_y = self.rect_xy_pos
                new_x, new_y = self.convert_rc_to_xy(rc_pos)
                dx, dy = new_x-old_x, new_y-old_y
                self.Canvas.move(self.rect, dx, dy)
                self.rect_xy_pos = (new_x, new_y)
        self.Canvas.create_text(x,y, text=str(Index), fill=colorText,)
        self.Canvas.update() 


    def convert_rc_to_xy(self, rc_pos):
        # 传入棋子在棋盘上的r、c值，传出其在canvas上的坐标位置
        SIDE = (435 - 400)/2
        DELTA = (400-2)/(self.boardWidth-1)
        r, c = rc_pos
        x = c*DELTA+SIDE
        y = r*DELTA+SIDE
        return x, y
    def convert_xy_to_rc(self, xy_pos):
        # 传入xy值，传出rc坐标
        SIDE = (435 - 400)/2
        DELTA = (400-2)/(self.boardWidth-1)
        x, y = xy_pos
        r = round((y-SIDE)/DELTA)
        c = round((x-SIDE)/DELTA)
        return r, c


    def selfPlay(self, player, Index=0):
        """ 
        构建一个AI自我博弈，重用搜索树，并存储自玩数据：(棋盘状态, 落子概率, 胜者预测) 以供训练
        """
        
        self.board.initBoard()
        boards, probs, currentPlayer = [], [], []

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
                        
                self.rect = None
                return winner, zip(boards, probs, winners_z)


    def humanMove(self, event):
        self.flag_human_click = True
        x, y = event.x, event.y
        r, c = self.convert_xy_to_rc((x, y))
        # 下一步棋
        self.move_human = r*self.boardWidth + c

    def playWithHuman(self, player):
        """ 
        与人对弈
        """
        self.Canvas.bind("<Button-1>", self.humanMove)
        # 先手玩家为 第0号玩家
        self.board.initBoard(0)

        KEY = 0
        while True:
            if self.board.current_player == 1:
                move, move_probs = player.getAction(self.board, self.flag_is_train)
                # 落子
                self.board.do_move(move)
                KEY = 1
            else:
                if self.flag_human_click:
                    if self.move_human in self.board.availables:
                        self.flag_human_click = False
                        self.board.do_move(self.move_human)
                        KEY = 1
                    else:
                        self.flag_human_click = False
                        print("无效区域")

            if self.flag_is_shown and KEY == 1:
                self.Show(self.board)
                KEY = 0

            gameOver, winner = self.board.gameIsOver()
            if gameOver:
                # 更新设置MCTS
                player.resetMCTS()
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
                # self.resetBoard()
                break
        self.rect = None