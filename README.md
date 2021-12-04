# Meta-Zeta
这是一个基于强化学习的自我对弈模型，运行之后的程序如下图所示。

<div align=center>
<img src="images/show-how.gif" width = "551" height = "357" align=center/>
</div>

## 运行代码
```
python3 MetaZeta.py
```
#### 1. 训练模型
我们构造一个基于 **MCTS** 的AI玩家，其中用 **残差神经网络** 辅助 **MCTS** 预测。    
操作：点击右上角单选`AI 自我对弈`，点击`开始`。

#### 2. 测试模型 
我们可以与训练好的AI对弈，测试AI的下棋水平  
操作：点击右上角单选`与 AI 对战`，点击`开始`。

## 文件结构
|文件名|类型|描述|     
|-|-|-|
|`TreeNode.py`|**MCTS**| MCTS 决策树的节点| 
|`MCTS.py`|**MCTS**|构建 MCTS 决策树|  
|`AIplayer.py`|**MCTS**|构建一个基于 MCTS+NN 的 AI|  
|`Board.py`|**Board**|存储棋盘信息| 
|`Game.py`|**Board**|定义了 selfPlay 以及 VS-Human 的游戏过程|  
|`PolicyNN.py`|**NN**|构建残差神经网络| 
|`MetaZeta.py`|**Main**|GUI综合各方 All in one| 

## 原理（附代码解释）
**持续更新 ···**
### 1. [棋盘设计](docs/Board.md)
### 2. [残差神经网络](docs/PolicyNN.md)
### 3. [MCTS](docs/MCTS.md)
### 4. [强化学习](docs/RL.md)

