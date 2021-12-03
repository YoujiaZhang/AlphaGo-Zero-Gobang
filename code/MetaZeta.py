from Board import *
from Game import *
from AIplayer import *
from PolicyNN import * 

import time
import os
from tensorflow.keras.utils import plot_model
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tkinter import *
from tkinter import scrolledtext
import tkinter as tk
import threading

class MetaZeta(threading.Thread):
    save_ParaFreq = 200 # 每过200盘自我对弈，就更新决策网络模型
    MAX_Games = 2000

    def __init__(self, flag_is_shown = True, flag_is_train = True):
        self.flag_is_shown = flag_is_shown
        self.flag_is_train = flag_is_train
        
        self.window = tk.Tk()
        self.window.resizable(0,0)
        self.window.title('Meta Zeta --- Youjia Zhang')
        self.window.geometry('810x500')

        self.btStart = tk.Button(self.window , text='开始', command=lambda :self.thredaTrain(self.train))
        self.btStart.place(x=480, y=10)
        
        self.btReset = tk.Button(self.window , text='重置', command = self.resetCanvas)
        self.btReset.place(x=540, y=10)

        self.iv_default = IntVar()
        self.rb_default1 = Radiobutton(self.window, text='AI 自我对弈', value=1, variable=self.iv_default)
        self.rb_default2 = Radiobutton(self.window, text='与 AI 对战', value=2,  variable=self.iv_default)
        self.rb_default1.place(x=595, y=15)
        self.rb_default2.place(x=695, y=15)

        self.canvas = tk.Canvas( self.window, bg='#CD853F', height=435, width=435)
        self.scrollText = scrolledtext.ScrolledText(self.window, width=35, height=24)

        # 构建棋盘
        self.game = Game(Canvas=self.canvas, scrollText=self.scrollText, flag_is_shown=self.flag_is_shown, flag_is_train=self.flag_is_train)
        # 构建神经网络
        self.NN = PolicyValueNet((4, self.game.boardWidth, self.game.boardHeight))

        if not self.flag_is_train:
            # self.NN.load_model("policy.model")
            self.NN = PolicyValueNet((4, self.game.boardWidth, self.game.boardHeight))
            self.drawScrollText("读取模型")

        # 构造MCTS玩家，将神经网络辅助MCTS进行决策
        self.MCTSPlayer = MCTSPlayer(policy_NN = self.NN.policy_NN)
        # 输出模型结构
        plot_model(self.NN.model, to_file='model.png', show_shapes=True)

        self.DrawCanvas((30, 30))
        self.DrawText((480, 50))
        self.DrawRowsCols((40, 470), (10, 35))

        self.window.mainloop()


    def thredaTrain(self, func,):
        # 将函数打包进线程
        myThread = threading.Thread(target=func,) 
        myThread.setDaemon(True) 
        myThread.start()


    def DrawCanvas(self, canvas_pos):
        x, y = canvas_pos
        # 画纵横线
        for i in range(self.game.boardWidth+1):
            pos = i*(400-2)/(self.game.boardWidth-1)
            SIDE = (435 - 400)/2
            self.canvas.create_line(SIDE, SIDE+pos, SIDE+400, SIDE+pos)
            self.canvas.create_line(SIDE+pos, SIDE, SIDE+pos, SIDE+400)
        self.canvas.place(x=x, y=y)

    def DrawRowsCols(self, rspos, cspos):
        rx, ry = rspos
        cx, cy = cspos
        for i in range(8):
            clabel = tk.Label(self.window, text=str(i))
            clabel.place(x=cx, y=cy+i*(400-2)/(self.game.boardWidth-1))

            rlabel = tk.Label(self.window, text=str(i))
            rlabel.place(x=rx+i*(400-2)/(self.game.boardWidth-1), y=ry)

    def DrawText(self, xy_pos):
        x, y = xy_pos
        self.scrollText.place(x=x, y=y)
    
    def drawScrollText(self, string):
        self.scrollText.insert(END, string+'\n')
        self.scrollText.see(END)
        self.scrollText.update()
    
    def resetCanvas(self):
        self.canvas.delete("all")
        self.scrollText.delete(1.0, END)
        self.DrawCanvas((30, 30))
        return

    def train(self):
        Loss = []
        if self.iv_default.get() == 1:
            self.flag_is_train = True
            self.game.flag_is_train = True
        else:
            self.flag_is_train = False
            self.game.flag_is_train = False

        # 总共进行 MAX_Games 场对弈
        for oneGame in range(self.MAX_Games):
            # start = time.time()
            if self.flag_is_train:
                # MCTS 进行自我对弈
                self.drawScrollText('正在 第'+str(oneGame+1)+'轮 自我对弈···')
                winner, play_data = self.game.selfPlay(self.MCTSPlayer,)

                # 为神经网络存储 训练数据
                self.NN.memory(play_data)

                # 如果数据池已经足够了《一批数据》的量，就对决策网络进行参数更新（训练）
                if len(self.NN.trainDataPool) > self.NN.trainBatchSize:
                    loss = self.NN.update(scrollText=self.scrollText)
                    Loss.append(loss)

                else:
                    self.drawScrollText("收集训练数据: %d%%" % (len(self.NN.trainDataPool)/self.NN.trainBatchSize*100))

                # 每过一定迭代次数保存模型
                if (oneGame+1) % self.save_ParaFreq == 0:
                    self.NN.save_model('policy.model')
                    self.drawScrollText("保存模型")
                
                self.canvas.delete("all")
                self.DrawCanvas((30, 30))
            else:
                self.game.playWithHuman(self.MCTSPlayer,)
            
            # 重置画布
            # end = time.time()
            # print("循环运行时间:%.2f秒"%(end-start))

if __name__ == '__main__':
    
    # if(len(sys.argv) == 2 and sys.argv[1] == "1"):
    #     flag_is_shown = False; flag_is_train = True
    # else:
    #     flag_is_shown = True; flag_is_train = False
    
    # tqdm(range(2,28-2),ncols=45)

    metaZeta = MetaZeta()
    # metaZeta