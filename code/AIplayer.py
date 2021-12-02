from MCTS import *

class MCTSPlayer():
    """
    基于MCTS的AI棋手
    """
    def __init__(self, policy_NN):
        self.simulations = 700  # 每次行动的模拟《simulation》数
        self.factor = 5         # factor 是一个从0到正无穷的调节因子 较高的值意味着更多地依赖于先验概率

        # mcts：就是AI棋手的《决策思维》
        # policy_NN : 预测网络
        self.mcts = MCTS(policy_NN, self.factor, self.simulations)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, flag_is_train):
        # 明智的行动
        sensible_moves = board.availables

        # move_probs 的尺寸是整个棋盘的大小
        # 每一个格子上存放着此处落子的概率
        move_probs = np.zeros(board.width * board.height)

        if len(sensible_moves) > 0:
            # 基于 MCTS 获取下一步的落子行为
            acts, probs = self.mcts.get_move_probs(board, flag_is_train)
            move_probs[list(acts)] = probs
            
            if flag_is_train:
                # 添加《Dirichlet Noise》进行探索（自我对弈训练所需）
                move = np.random.choice( # 随机抽取
                    acts, # 落子行为
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # 自下而上更新根节点并重用 MCTS
                self.mcts.update_with_move(move)

            else:
                # 非训练
                # ------------------------------------
                # 更新根节点并使用默认的temp=1e-3重用搜索树
                # 这几乎等同于选择prob最高的移动
                move = np.random.choice(acts, p=probs)
                # 重置根节点
                self.mcts.update_with_move(-1)

            return move, move_probs
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)