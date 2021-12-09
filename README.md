<h1 align="center">Meta-Zeta</h1>

<p align="center">
<img src="https://img.shields.io/badge/made%20by-youjiaZhang-blue.svg" >

<img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103" >
</p>

Do you like to play **Gobang** ?
Do you want to know how **AlphGo Zero** works ?
Check it out!

## View a Demo
This is a self-gaming model based on reinforcement learning, and the program after running is shown below.

<div align=center>
<img src="images/show-how.gif" width = "551" height = "357" align=center/>
</div>

---
## Quick Start
```
python3 MetaZeta.py
```
#### 1. Train
We construct an AI player based on **MCTS**, where **MCTS** prediction is aided by **residual neural network**.    
Operation: Click on `AI Self-Play` in the top right corner and click `Start`.

#### 2. Test
We can play against the trained AI to test the AI's chess playing level  
Action: Click `Play against AI` in the upper right corner and click `Start`.

#### 3. Environment
Ubuntu 18.04.6 LTS
tensorflow-gpu==2.6.2

---
## File Structure
|filename|type|description|     
|-|-|-|
|`TreeNode.py`|**MCTS**| nodes of the MCTS decision tree| 
|`MCTS.py`|**MCTS**|Build MCTS decision tree|  
|`AIplayer.py`|**MCTS**|Build an AI based on MCTS+NN|  
|`Board.py`|**Board**|store board information| 
|`Game.py`|**Board**|defines the game process for selfPlay and VS-Human|  
|`PolicyNN.py`|**NNN**|constructs a residual neural network| 
|`MetaZeta.py`|**Main**|GUI synthesis for all parties All in one| 

---
## Principle (with code explanation)
### 1. [Board design](docs/Board.md)
First we need to design some rules to describe the information on the board

### 2. [Residual Neural Network](docs/PolicyNN.md)
Then, we need to build a residual neural network ([Network structure](images/model.png))

### 3. [MCTS](docs/MCTS.md) 
Then, we need to understand how AI makes decisions. How he accumulates the learned knowledge of playing chess

### 4. [Reinforcement Learning](docs/RL.md)      
Finally we need to know the whole process of reinforcement learning (i.e. self-gaming)


