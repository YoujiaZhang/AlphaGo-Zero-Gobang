from TreeNode import *
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class MCTS():
    """
    蒙特卡树搜索树
    """

    def __init__(self, policy_NN, factor=5, simulations=10000):

        self._root = TreeNode(None, 1.0) # 初始化根节点
        self._policy = policy_NN       # 神经网络
        self._factor = factor            # factor 是一个从0到正无穷的调节因子 较高的值意味着更多地依赖于先验概率
        self._simulations = simulations  # 每次模拟推演《simulation》的数量

    def _playout(self, state):
        """
        推演： 从根到叶进行推演，在叶上获取一个值，并通过其父级将其传播回来。
        """
        node = self._root
        while True:
            # 如果，是叶子节点就跳出
            if node.is_leaf():
                break
            # 否则
            #   就选择该节点所有孩子中《分数》最高的。
            #       action：落子动作
            #       node：孩子节点
            action, node = node.select(self._factor)
            state.do_move(action)

        # ---------------------------------------------------
        # 在原本的MCTS中，《评估》操作采用蒙特卡洛的方法，通过随机下棋
        # 模拟走完一次完整棋局 (称为 rollout), 得到胜负结果。模拟的结
        # 果反应在 leaf_value 变量中
        # ---------------------------------------------------
        # 根据当前的《状态(棋盘)》使用神经网络预测：下一步所有的动作以及
        # 对应的概率 + 此步未来的收益
        action_probs, leaf_value = self._policy(state)

        # 检查一下当前《状态》是不是已经分出胜负
        end, winner = state.game_end()

        # 如果这盘棋还没结束
        if not end:
            # 扩展当前节点
            node.expand(action_probs)
        else:
            # 平局
            if winner == -1:
                leaf_value = 0.0
            else:
                # 如果《模拟/预测》的结果，获胜的是当前的玩家，+1分 否则 -1分
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # 根据神经网络的预测结果 leaf_value
        # 自下而上《更新》叶子
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, flag_is_train):
        """
        获取落子依据：按顺序进行所有推演，并返回可用操作及其相应的概率。
        """
        # state：当前游戏棋盘的状态。
        # 在（0，1）中控制探测(exploration)程序
        exploration = 1.0 if flag_is_train else 1e-3

        # 根据当前棋盘状态，经过 _simulations 次数的模拟
        # 构建出了一个 MCTS 树，根节点是依托于当前棋盘。
        for _ in range(self._simulations):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 根据根节点，获取下一步落子的决策
        act_visits = [(act, node._N_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/exploration * np.log(np.array(visits) + 1e-10))

        # 落子的决策
        return acts, act_probs

    def update_with_move(self, move):
        # 如果走的这一步在就是当前树根的孩子之一
        if move in self._root._children:
            # 更新树根
            self._root = self._root._children[move]
            self._root._parent = None
        else:
            # 否则保持原样
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"