import numpy as np

class TreeNode():
    """
    蒙特卡洛搜索树的节点类
    """
    def __init__(self, parent, prior_p):
        self._parent = parent # 父节点
        self._children = {}   # 孩子节点
        self._N_visits = 0    # 该节点的访问次数
        self._Q = 0           # 节点的总收益(V) / 总访问次数(N)
        self._U = 0           # 神经网络学习的目标：U 是正比于概率P，反比于访问次数N
        self._P = prior_p     # 走某一步棋(a)的先验概率
    
    def get_value(self, factor):
        """
        计算每个节点的《价值》，用以选择
        """
        # factor 是一个从0到正无穷的调节因子

        self._U = (factor * self._P *np.sqrt(self._parent._N_visits) / (1 + self._N_visits))
        return self._Q + self._U

    def expand(self, action_priors):
        """
        扩展：增加叶子节点的孩子
        """
        # action_priors：(动作a，概率p)

        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, factor):
        """
        选择：根据《策略》选择落子动作
        """
        # 选择所有孩子中《分数》最高
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(factor))

    def update(self, leaf_value):
        """
        更新：更新节点的数值
        """
        # leaf_value: 从当前选手的身份《评估》叶子节点的价值，也就是走这一步预计带来的收益。

        self._N_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._N_visits

    def update_recursive(self, leaf_value):
        """
        回溯：递归更新从叶子到根上的所有节点
        """
        # 如果这个节点是有父亲，优先更新该节点的父亲

        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)
    
    def is_leaf(self):
        """
        判断该节点是不是叶子节点：没有孩子的就是叶子
        """
        return self._children == {}

    def is_root(self):
        """
        判断该节点是不是根节点：没有父亲的就是根
        """
        return self._parent is None