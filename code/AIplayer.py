from MCTS import *

class MCTSPlayer():
    """
    基于MCTS的AI棋手
    """
    def __init__(self, policy_NN):
        self.simulations = 400  # 每次行动的模拟《simulation》数
        self.factor = 5         # factor 是一个从0到正无穷的调节因子 较高的值意味着更多地依赖于先验概率

        # MCTS：就是AI棋手的《决策思维》
        # policy_NN : 预测网络
        self.MCTS = MCTS(policy_NN, self.factor, self.simulations)

    def resetMCTS(self):
        self.MCTS.updateMCTS(-1)

    def getAction(self, board, flag_is_train):
        # 获得当前棋盘中可以落子的地方
        emptySpacesBoard = board.availables

        # move_probs 的尺寸是整个棋盘的大小
        # 每一个格子上存放着此处落子的概率
        move_probs = np.zeros(board.width * board.height)

        if len(emptySpacesBoard) > 0:
            # 基于 MCTS 获取下一步的落子行为，以及对应每一个位置胜率
            acts, probs = self.MCTS.getMoveProbs(board, flag_is_train)
            move_probs[list(acts)] = probs
            
            if flag_is_train:
                # 添加《Dirichlet Noise》进行探索（自我对弈训练所需）
                move = np.random.choice( # 随机抽取
                    acts, # 落子行为
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # 自下而上更新根节点并重用 MCTS
                # AI 相当于是《同一个人左右互搏》使用同一棵MCTS进行对我对弈
                self.MCTS.updateMCTS(move)

            else:
                # 非训练
                # ------------------------------------
                # 更新根节点并使用默认的temp=1e-3重用搜索树
                # 这几乎等同于选择prob最高的移动
                move = np.random.choice(acts, p=probs)
                # 重置 MCTS
                self.MCTS.updateMCTS(-1)
            
            # 依据概率选择下一步落子的位置，以及当前棋盘的所有位置的概率（分布）
            return move, move_probs
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)