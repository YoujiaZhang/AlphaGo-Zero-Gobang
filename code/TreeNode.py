import numpy as np

class TreeNode():
    """
    蒙特卡洛搜索树的节点类
    """
    def __init__(self, parent, prior_p):
        self.NUM = 1
        self.father = parent # 父节点
        self.children = {}   # 孩子节点
        self.N_visits = 0    # 该节点的访问次数
        self.Q = 0           # 节点的总收益(V) / 总访问次数(N)
        self.U = 0           # 神经网络学习的目标：U 是正比于概率P，反比于访问次数N
        self.P = prior_p     # 走某一步棋(a)的先验概率
    
    def getValue(self, factor):
        """
        计算每个节点的《价值》，用以选择
        """
        # factor 是一个从0到正无穷的调节因子
        #------------------------------------------------------------------------------
        # 如果 factor 越小，MCTS 搜索中的探索广度就越低，对神经网络输出的先验概       
        # 率的关注就越少，导致性能不理想。如果 factor 太大，探索广度就太高了，它太依
        # 赖于神经网络输出的先验概率。它不太重视 MCTS 模拟得到的结果，性能也不
        # 理想。因此，需要一个合理的 factor 值
        self.U = (factor * self.P *np.sqrt(self.father.N_visits) / (1 + self.N_visits))
        return self.Q + self.U

    def expand(self, action_priors):
        """
        扩展：增加叶子节点的孩子
        """
        # action_priors：(落子动作a，该位置的概率p)
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, factor):
        """
        选择：根据《策略》选择落子动作
        """
        # 选择所有孩子中《分数》最高
        return max(self.children.items(), key=lambda act_node: act_node[1].getValue(factor))

    def update(self, leaf_value):
        """
        更新：更新节点的数值
        """
        # leaf_value: 从当前选手的身份《评估》叶子节点的价值，也就是走这一步预计带来的收益。
        self.N_visits += 1
        self.Q += 1.0*(leaf_value - self.Q) / self.N_visits

    def updateRecursive(self, leaf_value):
        """
        回溯：递归更新从叶子到根上的所有节点
        """
        # 如果这个节点是有父亲，优先更新该节点的父亲
        if self.father:
            self.NUM = 0
            for i in list(self.children.items()):
                self.NUM += i[1].NUM
            self.father.updateRecursive(-leaf_value)
        self.update(leaf_value)
    
    def isLeaf(self):
        """
        判断该节点是不是叶子节点：没有孩子的就是叶子
        """
        return self.children == {}

    def isRoot(self):
        """
        判断该节点是不是根节点：没有父亲的就是根
        """
        return self.father is None
    
    def __str__(self):
        return "Node("+str(self.NUM)+','+str(len(self.children))+')'