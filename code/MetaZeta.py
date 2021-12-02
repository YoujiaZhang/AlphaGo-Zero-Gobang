from Board import *
from Game import *
from AIplayer import *
from PolicyNN import * 

import time
import os
from tensorflow.keras.utils import plot_model
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class MetaZeta():
    save_ParaFreq = 200 # 每过200盘自我对弈，就更新决策网络模型
    MAX_Games = 2000

    def __init__(self, flag_is_shown = False, flag_is_train = True):
        self.flag_is_shown = flag_is_shown
        self.flag_is_train = flag_is_train
        # 构建棋盘
        self.game = Game(self.flag_is_shown, self.flag_is_train)
        # 构建神经网络
        self.NN = PolicyValueNet((4, self.game.board_width, self.game.board_height))

        if not self.flag_is_train:
            self.NN.load_model("policy.model")
        # 构造MCTS玩家，将神经网络辅助MCTS进行决策
        self.MCTSPlayer = MCTSPlayer(policy_NN = self.NN.policy_NN)
        # 输出模型结构
        # plot_model(self.NN.model, to_file='model.png', show_shapes=True)


    def train(self):
        Loss = []
        winner = ''
        # 总共进行 MAX_Games 场对弈
        for oneGame in range(self.MAX_Games):
            start = time.time()
            if self.flag_is_train:
                # MCTS 进行自我对弈
                print('正在自我对弈···')
                winner, play_data = self.game.start_self_play(self.MCTSPlayer)
                print('对弈结束 胜者：玩家',winner)
                
                # 为神经网络存储 训练数据
                self.NN.memory(play_data)

                # 如果数据池已经足够了《一批数据》的量，就对决策网络进行参数更新（训练）
                if len(self.NN.trainDataPool) > self.NN.trainBatchSize:
                    loss = self.NN.update()
                    Loss.append(loss)
                else:
                    print("收集数据: %d%%, " % (len(self.NN.trainDataPool)/self.NN.trainBatchSize*100), end="")
                
                # 每过一定迭代次数保存模型
                if (oneGame+1) % self.save_ParaFreq == 0:
                    self.NN.save_model('policy.model')
                # print("oneGame = %d" % oneGame)
            else:
                self.game.start_play(self.MCTSPlayer)

            end = time.time()
            print("循环运行时间:%.2f秒"%(end-start))

import sys
if __name__ == '__main__':
    
    if(len(sys.argv) == 2 and sys.argv[1] == "1"):
        flag_is_shown = True; flag_is_train = True
    else:
        flag_is_shown = True; flag_is_train = False
        
    metaZeta = MetaZeta(flag_is_shown, flag_is_train)
    metaZeta.train()