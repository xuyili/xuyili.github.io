## 第三章

> 本章知识最好结合代码与实际问题进行实战。计划对船舶路径规划问题，实现本章涉及的多种算法，对比时间代价、内存开销和解的质量。
> 官方代码：https://github.com/aimacode/aima-python


环境是回合式的、单智能体的、完全可观测的、确定性的、静态的、离散的和已知的。算法需要在搜索所需时间、可用内存和解的质量之间进行权衡。如果我们对于启发式函数的形式拥有额外的领域相关知识来估计给定状态离目标有多远，或者我们预计算涉及模式或地标的部分解，算法会更高效。
* 在智能体开始搜索之前，必须形式化一个良定义的问题。
* 问题由 5 部分组成：初始状态、动作集合、描述这些动作结果的转移模型、目标状态集合和动作代价函数。
* 问题的环境用状态空间图表示。通过状态空间（一系列动作）从初始状态到达一个目标状态的路径是一个解。
* 搜索算法通常将状态和动作看作原子的，即没有任何内部结构（尽管我们在学习时引入了状态特征）。
* 根据完备性、代价最优性、时间复杂性和空间复杂性来评估搜索算法
* 无信息搜索方法只能访问问题定义。算法构建一棵搜索树，试图找到一个解。算法会根据其首先扩展的节点而有所不同。
    *  最佳优先搜索根据评价函数选择节点进行扩展。
    *  广度优先搜索首先扩展深度最浅的节点；它是完备的，对于单位动作代价是最优的，但具有指数级空间复杂性。
    *  一致代价搜索扩展路径代价 g(n) 最小的节点，对于一般的动作代价是最优的。
    *  深度优先搜索首先扩展最深的未扩展节点。它既不是完备的也不是最优的，但具有线性级空间复杂性。深度受限搜索增加了一个深度限制。
    *  迭代加深搜索在不断增加的深度限制上调用深度优先搜索，直到找到一个目标。当完成全部循环检查时，它是完备的，同时对于单位动作代价是最优的，且具有与广度优先搜索相当的时间复杂性和线性级空间复杂性。
    *  双向搜索扩展两个边界，一个围绕初始状态，另一个围绕目标，当两个边界相遇时搜索停止。
* 有信息搜索方法可以访问启发式函数 h(n) 来估计从 n 到目标的解代价。它们可以访问一些附加信息，例如，存有解代价的模式数据库。
    *  贪心最佳优先搜索扩展  h(n) 值最小的节点。它不是最优的，但通常效率很高。
    *  A* 搜索扩展 f(n) = g(n) + h(n) 值最小的节点。在 h(n) 可容许的条件下，A* 是完备的、最优的。对于许多问题来说，A* 的空间复杂性仍然很高。
    *  双向  A* 搜索有时比 A* 搜索本身更高效。
    *  IDA*（迭代加深  A* 搜索）是 A* 搜索的迭代加深版本，它解决了空间复杂性问题。 
    *  RBFS（递归最佳优先搜索）和 SMA*（简化的内存受限 A*）搜索是健壮的最优搜索算法，它们仅使用有限的内存；如果时间充足，它们可以解决对于 A* 来说内存不足的问题。
    *  束搜索限制了边界的大小；因此它是非完备的、次优的，但束搜索通常能找到相当好的解，运行速度也比完备搜索更快。
    *  加权A* 搜索将搜索专注于一个目标，以扩展更少的节点，但它牺牲了最优性。
* 启发式搜索算法的性能取决于启发式函数的质量。我们有时可以通过松弛问题定义、在模式数据库中存储预计算的子问题的解代价、定义地标点，或者从问题类的经验中学习来构建良好的启发式函数。


## 第三章 习题与答案

> Part II Problem-solving: 3. Solving Problems By Searching

### Exercise 1

:question:  **Question 1:**

解释为什么问题形式化必须在目标形式化之后。
> Explain why problem formulation must follow goal formulation.

:exclamation: **Answer 1:**
在目标形式化时，我们决定关注世界的哪些方面，而忽略或抽象掉什么方面。在问题形式化中，我们决定如何处理重要方面（并忽略其他方面）。如果我们首先形式化问题，我们将不知道要关注什么或忽视什么。也就是说，在目标形式化、问题形式化和问题解决之间存在一个迭代循环，直到找到一个足够有用和有效的解决方案为止


---

### Exercise 2

:question:  **Question 2:**

针对以下每个问题给出一个完整的问题公式。选择一个精确到足以实施的公式。

1. 六个玻璃盒子排成一排，每个盒子都有锁。前五个盒子中的每一个中都有一把钥匙，用于解开下一个顺序的盒子的锁；最后一个盒子里有一根香蕉。现在你有第一个盒子的钥匙，你想要得到那根香蕉。

2. 你从序列 ABABAECCEC 开始，或者任何以 A, B, C 和 E 组成的序列开始。你可以用以下等式转换这个序列：AC = E, AB = BC, BB = E, $Ex = x$ 对于任何$x$而言，你的目标是将序列转换成 E 序列，（即序列中只有 E 一个成员，就是这个序列：E）。举个例子，序列 ABBC 可以通过变换（BB=E）变为 AEC，然后变成 AC（通过变换 EC = C），最后变为目标序列 E（通过变换 AC = E）。

3. 有一个 $n \times n$ 的方格网格，每个方格最初要么是未涂漆的地板，要么是无底的坑。你开始站在一个未喷漆的地板方格上，然后可以在你下面的方格上喷漆，或者移动到相邻的未喷漆地板方格上。你要把整个地板都喷上漆。

4. 一艘满载集装箱的船在港口。有 13 排集装箱，每排 13 个集装箱宽，5 个集装箱高。你可以控制一台起重机，该起重机可以移动到船上方的任何位置，拾取其下方的集装箱，并将其移动到码头上。你想把船上的集装箱都卸下来。

> Give a complete problem formulation for each of the following problems. Choose a formulation that is precise enough to be implemented.
> 1. There are six glass boxes in a row, each with a lock. Each of the first five boxes holds a key unlocking the next box in line; the last box holds a banana. You have the key to the first box, and you want the banana.
> 2. You start with the sequence ABABAECCEC, or in general any sequence made from A, B, C, and E. You can transform this sequence using the following equalities: AC = E, AB = BC, BB = E, and Ex = x for any x . For example, ABBC can be transformed into AEC, and then AC, and then E. Your goal is to produce the sequence E.
> 3. There is an n×n grid of squares, each square initially being either unpainted floor or a bottomless pit. You start standing on an unpainted floor square, and can either paint the square under you or move onto an adjacent unpainted floor square. You want the whole floor painted.
>  4. A container ship is in port, loaded high with containers. There 13 rows of containers, each 13 containers wide and 5 containers tall. You control a crane that can move to any location above the ship, pick up the container under it, and move it onto the dock. You want the ship unloaded.

:exclamation: **Answer 2:**

1. 玻璃盒子
    1. 状态：盒子的开闭，钥匙的对错
    2. 初始状态：有第一个盒子的正确钥匙
    3. 动作：选择盒子，开锁
    4. 转移模型：若该钥匙能打开盒子，则从中取出新的钥匙，并使用新的钥匙尝试打开下一个盒子；若该钥匙不能打开盒子则用该钥匙尝试下一个盒子。
    5. 目标状态: 打开最后一个盒子
    6. 动作代价：尝试打开盒子的次数
    7. 状态空间为5的阶乘，每成功打开一个盒子，剩余待尝试的盒子数量都减一
2. 序列问题
    1. 状态：序列的字母数量，与字母种类
    2. 初始状态：初始序列
    3. 动作：选择哪两个字母组合进行等式转换
    4. 转移模型：若序列中有可以转换AC或BB，则优先将AC或BB转换为E；否则将AB转换为BC、或将BC转换为AB。
    5. 目标状态：结果序列最终只含有E
    6. 动作代价：等式转换的次数
    7. 状态空间：序列长度n
3. 网格问题
    1. 状态：方格的状态，人的位置
    2. 初始状态：位于网格中某一个方格内
    3. 动作：喷漆与移动
    4. 转移模型：若当前方格未喷漆，则喷漆并移动到相邻的不是无底的坑的方格；若当前方格已喷漆，则直接并移动到相邻的不是无底的坑的方格。
    5. 目标状态：所有地板都被喷漆
    6. 动作代价：移动次数
    7. 状态空间：$ n^2 $
4. 起重机问题
    1. 状态：集装箱数量与位置，起重机位置
    2. 初始状态：没有集装箱被拾取，起重机位于任意位置
    3. 动作：选择集装箱，拾取并移动到码头上
    4. 转移模型：若起重机下方有集装箱，则拾取；若起重机下方没有集装箱则移动到其他位置
    5. 目标状态：船上集装箱的数量为0
    6. 动作代价：拾取集装箱，移动集装箱
    7. 状态空间：$13 \times 13 \times 5$
    
---

### Exercise 3

:question: **Question 3:**

你的目标是引导机器人走出迷宫。机器人最开始在迷宫的中心，并且面朝北方。你可以将机器人转向北、东、南或西。您可以指示机器人向前移动一定距离，但它会在撞到墙壁之前停下来。

1. 形式化这个问题。状态空间有多大？
2. 在走迷宫时，我们唯一需要转弯的地方是两条或多条走廊的交汇处。使用这个观察重新形式化这个问题。现在状态空间有多大？
3. 从迷宫中的每个点，我们可以向四个方向中的任何一个方向移动，直到到达一个转折点，这是我们唯一需要做的动作。使用这些动作重新形式化问题。我们现在需要跟踪机器人的方向吗？
4. 在我们对问题的最初描述中，我们已经从现实世界中抽象出来，限制了动作并删除了细节。列出我们所做的三个简化。

> Your goal is to navigate a robot out of a maze. The robot starts in the center of the maze facing north. You can turn the robot to face north, east, south, or west. You can direct the robot to move forward a certain distance, although it will stop before hitting a wall.

>  1. Formulate this problem. How large is the state space?
>  2. In navigating a maze, the only place we need to turn is at the intersection of two or more corridors. Reformulate this problem using this observation. How large is the state space now?
>  3. From each point in the maze, we can move in any of the four directions until we reach a turning point, and this is the only action we need to do. Reformulate the problem using these actions. Do we need to keep track of the robot’s orientation now?
>  4. In our initial description of the problem we already abstracted from the real world, restricting actions and removing details. List three such simplifications we made.

:exclamation: **Answer 3:**

1. 形式化这个问题。状态空间有多大？
    1. 状态：机器人位置和朝向的状态描述。
    2. 初始状态：在迷宫中心，面朝北方。
    3. 动作：向前移动一定距离d；改变机器人的方向。
    4. 转移模型：将状态和动作映射为一个结果状态。
    5. 目标状态: 机器人在出口处。
    6. 动作代价：移动的距离。
    7. 状态空间是无限大的，因为机器人可以处在任何位置。
2. 在走迷宫时，我们唯一需要转弯的地方是两条或多条走廊的交汇处。使用这个观察重新形式化这个问题。现在状态空间有多大？
    1. 状态：机器人当前所处的交汇处，以及它所面对的方向
    2. 初始状态：在迷宫中心，面朝北方。
    3. 动作：移动到前面的下一个交汇处（如果存在）。转向新的方向。
    4. 转移模型：将状态和动作映射为一个结果状态。
    5. 目标状态：机器人在出口处。
    6. 动作代价：移动的距离。
    7. 状态空间有 4n 个状态，其中 n 是交叉点的数量。
3. 我们现在需要跟踪机器人的方向吗？
    1. 状态：机器人当前所处的交汇处，以及它所面对的方向
    2. 初始状态：在迷宫中心，面朝北方。
    3. 动作：移动到北、南、东或西的下一个交汇处。
    4. 转移模型：将状态和动作映射为一个结果状态。
    5. 目标状态: 机器人在出口处。
    6. 动作代价：移动的距离。
    7. 状态空间有 4n 个状态，其中 n 是交叉点的数量。
    8. 不再需要跟踪机器人的方向，因为它与预测我们行动的结果无关，也不是目标的一部分。
4. 列出我们所做的三个简化。
    1. 状态抽象：
        * 忽略机器人离地面的高度，不管它是否偏离垂直方向。
        * 机器人只能面向四个方向。
        * 世界其他地区被忽视：迷宫中其他机器人的可能性，环境对机器人运动影响程度。
    2. 行动抽象：
        * 我们假定了所有位置都可以安全到达：机器人不会被卡住或损坏。
        * 机器人可以随心所欲地移动，而无需重新进行充电。
        * 简化的移动系统：向前移动一定距离，而不是控制每个单独的电机并观察传感器以检测碰撞。

--- 

### Exercise 4
:question: **Question 4:**

你有一个 $9 \times 9$ 方格网格，每个方格都可以是红色或蓝色的。网格最初都是蓝色的，但你可以多次改变任何正方形的颜色。想象一下，网格被划分为 9 个 $3 \times 3$ 的子方格，你希望每个子方格都是一种颜色，但相邻的子方格是不同的颜色。

1. 用直截了当的方法表述这个问题。计算状态空间的大小。
2. 你只能给一个方格上色一次。重新表述，并计算状态空间的大小。广度优先搜索在这个问题上会比在 (1) 中执行得更快吗？那么迭代加深树搜索呢？
3. 给定目标，我们只需要考虑每个子方块都是统一着色的颜色。重新表述问题并计算状态空间的大小。
4. 这个问题有多少个解？
5. (2) 和 (3) 部分先后抽象了原问题 (1) ，你能否将问题 (3) 的解转化为问题 (2) 的解，并将问题 (2) 的解转化为问题 (1) 的解？

:exclamation: **Answer 4:**

