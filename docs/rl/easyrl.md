# EasyRL强化学习

> 正在梳理中，持续更新

## 准备

[DataWhale蘑菇书EasyRL](https://datawhalechina.github.io/easy-rl/#/)

[强化学习纲要github](https://github.com/zhoubolei/introRL)

[世界冠军带你从零实践强化学习](https://aistudio.baidu.com/aistudio/education/group/info/1335)

[李宏毅《强化学习》](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)

## 第一章 概述

### 思维导图

![RL思维导图](https://docs-xy.oss-cn-shanghai.aliyuncs.com/rl01.png)


### 关键词概念

- 强化学习（Reinforcement Learning）：Agent可以在与复杂且不确定的Environment进行交互时，尝试使所获得的Reward最大化的计算算法。
- Action： Environment接收到的Agent当前状态的输出。
- State：Agent从Environment中获取到的状态。
- Reward：Agent从Environment中获取的反馈信号，这个信号指定了Agent在某一步采取了某个策略以后是否得到奖励。
- Exploration：在当前的情况下，继续尝试新的Action，其有可能会使你得到更高的这个奖励，也有可能使你一无所有。
- Exploitation：在当前的情况下，继续尝试已知的可以获得最大Reward的过程，即重复执行这个 Action 就可以了。
- 深度强化学习（Deep Reinforcement Learning）：不需要手工设计特征，仅需要输入State让系统直接输出Action的一个end-to-end training的强化学习方法。通常使用神经网络来拟合 value function 或者 policy network。
- Full observability、fully observed和partially observed：当Agent的状态跟Environment的状态等价的时候，我们就说现在Environment是full observability（全部可观测），当Agent能够观察到Environment的所有状态时，我们称这个环境是fully observed（完全可观测）。一般我们的Agent不能观察到Environment的所有状态时，我们称这个环境是partially observed（部分可观测）。
- POMDP（Partially Observable Markov Decision Processes）：部分可观测马尔可夫决策过程，即马尔可夫决策过程的泛化。POMDP 依然具有马尔可夫性质，但是假设智能体无法感知环境的状态 s，只能知道部分观测值 o。
- Action space（discrete action spaces and continuous action spaces）：在给定的Environment中，有效动作的集合经常被称为动作空间（Action space），Agent的动作数量是有限的动作空间为离散动作空间（discrete action spaces），反之，称为连续动作空间（continuous action spaces）。
- policy-based（基于策略的）：Agent会制定一套动作策略（确定在给定状态下需要采取何种动作），并根据这个策略进行操作。强化学习算法直接对策略进行优化，使制定的策略能够获得最大的奖励。
- valued-based（基于价值的）：Agent不需要制定显式的策略，它维护一个价值表格或价值函数，并通过这个价值表格或价值函数来选取价值最大的动作。
- model-based（有模型结构）：Agent通过学习状态的转移来采取措施。
- model-free（无模型结构）：Agent没有去直接估计状态的转移，也没有得到Environment的具体转移变量。它通过学习 value function 和 policy function 进行决策。

### 知识点总结

强化学习与监督学习的不同之处：
> 1. 输入的是**序列数据**，而不是像监督学习那样满足IID（独立同分布）的样本
> 2. 学习者（learner）没有被告诉什么是正确的行为，他必须自己**尝试并发现哪些行为可以获得奖励**
> 3. 智能体（agent）在获得能力的方法是不断地进行**试错探索（Trial-and-error exploration）**
> 4. 强化学习中没有监督者（supervisor），只有奖励信号（reward signal），并且**奖励是延迟的**

强化学习的特征：
> 1. 试错探索（Trial-and-error exploration）：需要通过探索Environment来获取对这个Environment的理解。
> 2. 延迟奖励（Delayed reward）：Agent会从Environment里面获得延迟的Reward。
> 3. 时间关联（Time matters）：训练过程中时间非常重要，因为数据都是有时间关联的，而不是像监督学习一样是IID分布的。
> 4. 智能体行为影响我们采集到的序列数据（智能体行为改变环境）：Agent的Action会影响它随后得到的反馈。

强化学习的基本结构：

> **本质上是Agent和Environment间的交互**。当Agent在Environment中得到当前时刻的State，Agent会基于此状态输出一个Action。然后这个Agent会加入到Environment中去并输出下一个State和当前的这个Action得到的Reward。Agent在Environemnt里面存在的目的就是为了最大化它的期望的累计奖励。

状态和观测有什么关系：

> **状态（State）**是对世界的**完整描述**，不会隐藏世界的信息。
> **观测（Observation）**是对状态的**部分描述**，可能会遗漏一些信息。
> 在深度强化学习中，我们几乎总是用一个实值向量、矩阵或者更高阶的张量来表示状态和观测。

对于一个强化学习的Agent，它由什么组成：

> 1. **策略函数（policy function）**：Agent会用这个函数来选取它下一步的动作，包括**随机性策略（stochastic policy）**和**确定性策略（deterministic policy）**。
> 2. **价值函数（value function）**：我们用价值函数来对当前状态进行评估，即进入现在的状态，到底可以对你后面的收益带来多大的影响。当这个价值函数大的时候，说明你进入这个状态越有利。
> 3. **模型（model）**：其表示了 Agent 对这个Environment的状态进行的理解，它决定了这个系统是如何进行的。

策略是什么:

> **策略**是智能体的动作模型，它决定了智能体的动作。它其实是一个函数，用于把输入的状态变成动作。**策略可分为两种：随机性策略和确定性策略**。
> **随机性策略（stochastic policy）**就是 π 函数，即输入一个状态 s，输出一个概率。 这个概率是智能体所有动作的概率，然后对这个概率分布进行采样，可得到智能体将采取的动作。比如可能是有 0.7 的概率往左，0.3 的概率往右，那么通过采样就可以得到智能体将采取的动作。
> **确定性策略（deterministic policy）**就是智能体直接采取最有可能的动作

根据强化学习 Agent 的不同，我们可以将其分为哪几类？

> 1. **基于价值函数的Agent（value-based agent）**： 显式学习的就是价值函数，隐式的学习了它的策略。因为这个策略是从我们学到的价值函数里面推算出来的。
> 2. **基于策略的Agent（policy-based agent）**：agent直接学习它的 policy，对agent输入state，将会输出这个动作（状态）的概率。policy-based agent没有去学习它的价值函数。
> 3. **演员-评论员智能体（actor-critic agent）**：把 value-based 和 policy-based 结合起来就有了 Actor-Critic agent。这一类 Agent 就把它的策略函数和价值函数都学习了，然后通过两者的交互得到一个更佳的状态。

基于策略迭代和基于价值迭代的强化学习方法有什么区别？
> 1. 基于策略迭代的强化学习方法，agent会制定一套动作策略（确定在给定状态下需要采取何种动作），并根据这个策略进行操作。强化学习算法直接对策略进行优化，使制定的策略能够获得最大的奖励；基于价值迭代的强化学习方法，agent不需要制定显式的策略，它维护一个价值表格或价值函数，并通过这个价值表格或价值函数来选取价值最大的动作。
> 2. **基于价值迭代的方法只能应用在不连续的、离散的环境下**（如围棋或某些游戏领域），对于行为集合规模庞大、动作连续的场景（如机器人控制领域），其很难学习到较好的结果（此时**基于策略迭代的方法能够根据设定的策略来选择连续的动作**）
> 3. 基于价值迭代的强化学习算法有Q-learning、Sarse等，而基于策略迭代的强化学习算法有策略梯度算法等。
> 4. 此外，Actor-Critic算法同时使用策略和价值评估来做出决策，其中，agent会根据policy做出state，而价值函数会对做出的state给出value，这样可以在原有的策略梯度算法的基础上加速学习过程，取得更好的效果。


有模型（model-based）学习和免模型（model-free）学习有什么区别？
> **针对是否需要对真实环境建模，强化学习可以分为有模型学习和免模型学习。**
> 有模型学习是指根据环境中的经验，构建一个虚拟世界，同时在真实环境和虚拟世界中学习；免模型学习是指不对环境进行建模，直接与真实环境进行交互来学习到最优策略。
> 总的来说，有模型学习相比于免模型学习仅多出了一个步骤，即对真实环境进行建模。免模型学习通常属于数据驱动型方法，需要大量的采样来估计状态、动作及奖励函数，从而优化动作策略。
> 免模型学习的泛化性要优于有模型学习，原因是有模型学习需要对真实环境进行建模，并且虚拟世界与真实环境之间可能还有差异，这限制了有模型学习算法的泛化性。

强化学习的通俗理解:
> environment 跟 reward function 不是我们可以控制的，environment 跟 reward function 是在开始学习之前，就已经事先给定的。我们唯一能做的事情是调整 actor 里面的 policy，使得 actor 可以得到最大的 reward。Actor 里面会有一个 policy， 这个 policy 决定了actor的行为。Policy 就是给一个外界的输入，然后它会输出 actor 现在应该要执行的行为。


### 面试题

看来你对于RL还是有一定了解的,那么可以用一句话谈一下你对于强化学习的认识吗?
> 答: 强化学习包含环境，动作和奖励三部分，其本质是agent通过与环境的交互，使得其作出的action所得到的决策得到的总的奖励达到最大，或者说是期望最大。

你认为强化学习与监督学习和无监督学习有什么区别?
> 答: 首先强化学习和无监督学习是不需要标签的，而监督学习需要许多有标签的样本来进行模型的构建；对于强化学习与无监督学习，无监督学习是直接对于给定的数据进行建模，寻找数据(特征)给定的隐藏的结构，一般对应的聚类问题，而强化学习需要通过延迟奖励学习策略来得到"模型"对于正确目标的远近(通过奖励惩罚函数进行判断)，这里我们可以将奖励惩罚函数视为正确目标的一个稀疏、延迟形式。另外强化学习处理的多是序列数据，样本之间通常具有强相关性，但其很难像监督学习的样本一样满足IID条件。

根据你上面介绍的内容，你认为强化学习的使用场景有哪些呢?
> 答: 七个字的话就是多序列决策问题。或者说是对应的模型未知，需要通过学习逐渐逼近真实模型的问题并且当前的动作会影响环境的状态,即服从马尔可夫性的问题。同时应满足所有状态是可重复到达的(满足可学习型的)。

强化学习中所谓的损失函数与DL中的损失函数有什么区别呀?
> 答: DL中的loss function目的是使预测值和真实值之间的差距最小，而RL中的loss function是是奖励和的期望最大。

你了解model-free和model-based吗?两者有什么区别呢?
> 答: 两者的区别主要在于是否需要对于真实的环境进行建模，model-free不需要对于环境进行建模，直接与真实环境进行交互即可，所以其通常需要较大的数据或者采样工作来优化策略，这也帮助model-free对于真实环境具有更好的泛化性能；而model-based 需要对于环境进行建模，同时再真实环境与虚拟环境中进行学习，如果建模的环境与真实环境的差异较大，那么会限制其泛化性能。现在通常使用model-free进行模型的构建工作。

### 实验

实验代码仓库：https://github.com/johnjim0816/rl-tutorials
环境准备：Python 3.7、PyTorch 1.10.0、Gym 0.21.0

#### install pytorch v1.10.0

conda : Linux and Windows
```bash
# CUDA 10.2
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

pip : Linux and Windows
```bash
# CUDA 11.3
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

#### install Gym v0.21.0

```bash
pip install gym==0.21.0
pip install ipykernel==6.15.1
pip install jupyter==1.0.0
pip install matplotlib==3.5.2
pip install seaborn==0.11.2
pip install dill==0.3.5.1
pip install argparse==1.4.0
```

#### 仓库使用说明

对于codes：
- 运行带有task的py脚本
对于Jupyter Notebook：
- 直接运行对应的ipynb文件

## 第二章 马尔可夫决策过程

### 关键词概念

- 马尔可夫性质(Markov Property): 如果某一个过程未来的转移跟过去是无关，只由现在的状态决定，那么其满足马尔可夫性质。换句话说，一个状态的下一个状态只取决于它当前状态，而跟它当前状态之前的状态都没有关系。
- 马尔可夫链(Markov Chain): 概率论和数理统计中具有马尔可夫性质（Markov property）且存在于离散的指数集（index set）和状态空间（state space）内的随机过程（stochastic process）。
- 状态转移矩阵(State Transition Matrix): 状态转移矩阵类似于一个 conditional probability，当我们知道当前我们在$s_{t}$这个状态过后，到达下面所有状态的一个概念，它每一行其实描述了是从一个节点到达所有其它节点的概率。
- 马尔可夫奖励过程(Markov Reward Process, MRP)：即马尔可夫链再加上了一个奖励函数。在 MRP之中，转移矩阵跟它的这个状态都是跟马尔可夫链一样的，多了一个奖励函数(reward function)。奖励函数是一个期望，它说当你到达某一个状态的时候，可以获得多大的奖励。
- horizon: 定义了同一个 episode 或者是整个一个轨迹的长度，它是由有限个步数决定的。
- return: 把奖励进行折扣(discounted)，然后获得的对应的收益。
- Bellman Equation（贝尔曼等式）：定义了当前状态与未来状态的迭代关系，表示当前状态的值函数可以通过下个状态的值函数来计算。Bellman Equation 因其提出者、动态规划创始人 Richard Bellman 而得名 ，同时也被叫作“动态规划方程”。$V(s)=R(s)+\gamma{\textstyle \sum_{s'\in S}} P(s'|s)V(s')$，特别地，矩阵形式：$V=R+\gamma PV$。
- Monte Carlo Algorithm（蒙特卡洛方法）：可用来计算价值函数的值。通俗的将，我们当得到一个MRP过后，我们可以从某一个状态开始，然后让它把这个小船放进去，让它随波逐流，这样就会产生一个轨迹。产生了一个轨迹过后，就会得到一个奖励，那么就直接把它的Discounted的奖励$g$直接算出来。算出来过后就可以把它积累起来，当积累到一定的轨迹数量过后，然后直接除以这个轨迹，就会得到它的这个价值。
- Iterative Algorithm（动态规划方法）： 可用来计算价值函数的值。通过一直迭代对应的Bellman Equation，最后使其收敛。当这个最后更新的状态跟你上一个状态变化并不大的时候，这个更新就可以停止。
- Q函数 (action-value function)：其定义的是某一个状态某一个行为，对应的它有可能得到的 return 的一个期望（over policy function）。
- MDP中的prediction（即policy evaluation问题）： 给定一个 MDP 以及一个 policy$\pi$，去计算它的value function，即每个状态它的价值函数是多少。其可以通过动态规划方法（Iterative Algorithm）解决。
- MDP中的control问题：寻找一个最佳的一个策略，它的input就是MDP，输出是通过去寻找它的最佳策略，然后同时输出它的最佳价值函数（optimal value function）以及它的这个最佳策略（optimal policy）。optimal policy使得每个状态，它的状态函数都取得最大值。所以当我们说某一个MDP的环境被解了过后，就是说我们可以得到一个optimal value function，然后我们就说它被解了。

### 知识点总结

为什么在马尔可夫奖励过程（MRP）中需要有discount factor?
> 1. 首先，有些马尔可夫过程是**带环**的，它并没有终结，我们想要**避免无穷的奖励**；
> 2. 另外，我们是想把这个不确定性也表示出来，希望尽可能快地得到奖励，而不是在未来某一个点得到奖励；
> 3. 接上面一点，如果这个奖励它是有实际价值的了，我们可能是更希望立刻就得到奖励，而不是我们后面再得到奖励；
> 4. 还有在有些时候，这个系数也可以把它设为0。比如说，当我们设为0过后，然后我们就只关注了它当前的奖励。我们也可以把它设为1，设为1的话就是对未来并没有折扣，未来获得的奖励跟我们当前获得的奖励是一样的。
> 
> 所以，这个系数其实是应该可以作为强化学习agent的一个hyperparameter来进行调整，然后就会得到不同行为的agent。

为什么矩阵形式的Bellman Equation的解析解比较难解？
> 通过矩阵求逆的过程，就可以把这个V的这个价值的解析解直接求出来。但是一个问题是这个矩阵求逆的过程的复杂度是$O(N^3)$。所以就当我们状态非常多的时候，比如当我们有一百万个状态时，转移矩阵会是一个一百万乘以一百万的一个矩阵。这样一个大矩阵的求逆是非常困难的，所以这种通过解析解去解，只能用于很小量的MRP

计算贝尔曼等式（Bellman Equation）的常见方法以及区别？
> 1. Monte Carlo Algorithm （蒙特卡洛方法）：可用来计算价值函数的值。通俗的讲，我们当得到一个MRP过后，我们可以从某一个状态开始，然后让它把这个小船放进去，让它随波逐流，这样就会产生一个轨迹。产生了一个轨迹过后，就会得到一个奖励，那么就直接把它的Discounted的奖励$g$直接算出来。算出来过后就可以把它积累起来，当积累到一定的轨迹数量过后，然后直接除以这个轨迹，就会得到它的这个价值。
> 2. Iterative Algorithm（动态规划方2. ）： 可用来计算价值函数的值。通过一直迭代对应的Bellman Equation，最后使其收敛。当这个最后更新的状态跟你上一个状态变化并不大的时候，**通常是小于一个阈值$\gamma$**，这个更新就可以停止。
> 3. 以上两者的结合方法：另外我们也可以通过Temporal-Difference Learning的方法。也叫**TD Learning**，是动态规划和蒙特卡洛方法的一个结合

马尔可夫奖励过程（MRP）与马尔可夫决策过程（MDP）的区别？
> 相对于MRP，马尔可夫决策过程多了一个decision，其他的定义与MRP都类似。这里我们多了一个决策，多了一个action，那么这个状态转移也多了一个condition。即采取某一种行为，你未来的状态也会不同。它不仅是依赖于你当前的状态，也依赖于在当前状态你这个agent，agent采取的这个行为会决定它未来的这个状态走向。对于这个价值函数，它也是多了一个条件，多了一个你当前的这个行为，就是说你当前的状态以及你采取的行为会决定你当前可能得到多少奖励。
> 另外，两者之间是有转换关系的。具体来说，已知一个MDP以及一个policy$\pi$的时候，我们可以把MDP转换成MRP。在MDP里面，转移函数$P(s'|s,a)$是基于它当前状态以及它当前的action，因为我们现在已知它policy function，就是说在每一个状态，我们知道它可能采取的行为的概率

MDP里面的状态转移跟MRP以及MP的结构或者计算方面的差异？
> - **马尔可夫链**的转移是直接就决定。**从你当前是什么状态，直接通过转移概率就直接决定了你下一个状态会是什么。**
> - 但是对于**MDP，它中间多了一层行为action**。在当前这个状态，你首先要决定的是采取某一种行为。然后因为你有一定的不确定性，在当前状态决定你当前采取的行为过后，你到未来的状态其实也是一个概率分布。在此概率分布下多的这一层action，是说你有多大的概率到达某一个未来状态，或者说你有多大的概率到达另外一个状态。所以当前状态与未来状态的转移过程中多了一层决策性，这是MDP和之前的马尔可夫过程不同的一个地方。**多了一个component，agent会采取行为来决定未来的状态转移。**

我们如何寻找最佳的policy，方法有哪些？
> 本质来说，当我们取得最佳的价值函数过后，我们可以通过对这个Q函数进行极大化，然后得到最佳的价值。然后，我们直接在这个Q函数上取一个让这个action最大化的值，我们就可以直接提取出它的最佳的policy。
> 具体方法：
>   - 穷举法（一般不使用）：假设我们有有限多个状态、有限多个行为可能性，那么每个状态我们可以采取这个 A 种行为的策略，那么总共就是$|A|^{|S|}$个可能的policy。我们可以把这个穷举一遍，然后算出每种策略的value function，然后对比一下可以得到最佳策略。这种方法效率极低。
>  - **Policy iteration**：一种迭代方法，有两部分组成，下面两个步骤一直在迭代进行，最终收敛：（有些类似于ML中EM算法（期望-最大化算法））
>    - 第一个步骤是**policy evaluation**，即当前我们在优化这个policy$\pi$，所以在优化过程中得到一个最新的policy。
>    - 第二个步骤是**policy improvement**，即取得价值函数后，进一步推算出它的Q函数。得到Q函数过后，那我们就直接去取它的极大化。
>  - Value iteration：我们一直去迭代Bellman Optimality Equation，到了最后，它能逐渐趋向于最佳的策略，这是value iteration算法的精髓，就是我们去为了得到最佳的$v^*$，对于每个状态它的$v^*$这个值，我们直接把这个Bellman Optimality Equation进行迭代，迭代了很多次之后它就会收敛至最佳的policy以及其对应的状态，这里面是没有policy function的。

请问马尔可夫过程是什么?马尔可夫决策过程又是什么?其中马尔可夫最重要的性质是什么呢?
> 马尔可夫过程是一个二元组$<S,P>$，$S$为状态的集合，$P$为状态转移概率矩阵；而马尔可夫决策过程是一个五元组$<S,P,A,R,\gamma>$，其中R表示为从$S$到$S'$能够获得的奖励期望，$\gamma$为折扣因子，$A$为动作集合。
> **马尔可夫最重要的性质是下一个状态只与当前状态有关，与之前的状态无关，也就是$P[S_{t+1}|S_t]=P[S_{t+1}|S_1,S_2,...,S_t]$**

请问我们一般怎么求解马尔可夫决策过程？
> 我们直接求解马尔可夫决策过程可以**直接求解贝尔曼等式（动态规划方程）**，即$V(s)=R(s)+\gamma{\textstyle \sum_{s'\in S}} P(s'|s)V(s')$，特别地，矩阵形式：$V=R+\gamma PV$。但是贝尔曼等式很难求解且计算复杂度较高，所以可以使用动态规划，蒙特卡洛，时间差分等方法求解。

请问如果数据流不满足马尔可夫性怎么办？应该如何处理？
> 如果不满足马尔可夫性，即下一个状态与之前的状态也有关，若还仅仅用当前的状态来进行求解决策过程，势必导致决策的泛化能力变差。为了解决这个问题，可以**利用RNN对历史信息建模，获得包含历史信息的状态表征**。表征过程可以使用注意力机制等手段。最后在表征状态空间求解马尔可夫决策过程问题。

请分别写出基于状态值函数的贝尔曼方程以及基于动作值的贝尔曼方程。
> 基于状态值函数的贝尔曼方程
> $$v_\pi(s)= {\textstyle \sum_{a}^{}} π(a∣s){\textstyle \sum_{s',r}}p(s',r∣s,a)[r(s,a)+\gamma v_\pi(s')]$$
> 基于动作值的贝尔曼方程
> $$q_\pi(s,a)={\textstyle \sum_{s',r}}p(s',r∣s,a)[r(s',a)+\gamma v_\pi(s')]$$

请问最佳价值函数（optimal value function）$v^*$和最佳策略（optimal policy）$\pi^*$为什么等价呢？
> 最佳价值函数的定义为：$v^*(s)=max_\pi v^{\pi}(s)$即我们去搜索一种policy$\pi$来让每个状态的价值最大。$v^*$就是到达每一个状态，它的值的极大化情况。在这种极大化情况上面，我们得到的策略就可以说是它的最佳策略（optimal policy），即$\pi^*(s)=\argmax_av^{\pi}(s)$。最佳策略是的每个状态的价值函数都取得最大值。所以如果我们可以得到一个optimal value function，就可以说某一个MDP的环境被解。在这种情况下，它的最佳的价值函数是一致的，就它达到的这个上限的值是一致的，但这里可能有多个最佳的policy，就是说多个policy可以取得相同的最佳价值。

能不能手写一下第n不的值函数更新公式？另外，当n越来越大时，值函数的期望和方差分别变大还是变小？
> n越大，方差越大，期望偏差越小。值函数的更新公式：
> $$Q(S,A) \leftarrow Q(S,A)+\alpha[\sum_{i=1}^{n}\gamma^{i-1}R_{t+i}+\gamma^n\max_aQ(S',a)-Q(S,A)]$$

## 第三章 表格型方法

### 关键词概念

- P函数和R函数： P函数反应的是状态转移的概率，即反应的环境的随机性，R函数就是Reward function。但是我们通常处于一个未知的环境（即P函数和R函数是未知的）。
- Q表格型表示方法： 表示形式是一种表格形式，其中横坐标为 action（agent）的行为，纵坐标是环境的state，其对应着每一个时刻agent和环境的情况，并通过对应的reward反馈去做选择。一般情况下，Q表格是一个已经训练好的表格，不过，我们也可以每进行一步，就更新一下Q表格，然后用下一个状态的Q值来更新这个状态的Q值（即时序差分方法）。
- 时序差分（Temporal Difference）： 一种Q函数（Q值）的更新方式，也就是可以拿下一步的Q值$Q(S_{t+1},A_{t+1})$来更新我这一步的Q值$Q(S_t,A_t)$。完整的计算公式如下：$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1}+\gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]$
- SARSA算法： 一种更新前一时刻状态的单步更新的强化学习算法，也是一种on-policy策略。该算法由于每次更新值函数需要知道前一步的状态(state)，前一步的动作(action)、奖励(reward)、当前状态(state)、将要执行的动作(action)，即$(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$这几个值，所以被称为SARSA算法。agent每进行一次循环，都会用$(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$对前一步的Q值（函数）进行一次更新

### 知识点总结

构成强化学习马尔可夫决策过程的四元组有哪些变量？
> - 答：状态、动作、状态转移概率和奖励，分别对应（S，A，P，R），后面有可能会加上个衰减因子构成五元组。

基于以上的描述所构成的强化学习的<学习>流程。
> - 答：强化学习要像人类一样去学习了，人类学习的话就是一条路一条路的去尝试一下，先走一条路，我看看结果到底是什么。多试几次，只要能一直走下去的，我们其实可以慢慢的了解哪个状态会更好。我们用价值函数$V(s)$来代表这个状态是好的还是坏的。然后用这个Q函数来判断说在什么状态下做什么动作能够拿到最大奖励，我们用Q函数来表示这个状态-动作值。

基于SARSA算法的agent的学习过程。
> - 答：我们现在有环境，有agent。每交互一次以后，我们的agent会向环境输出action，接着环境会反馈给agent当前时刻的state和reward。那么agent此时会实现两个方法：
>   1. 使用已经训练好的Q表格，对应环境反馈的state和reward选取对应的action进行输出。
>   2. 我们已经拥有了$(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$这几个值，并直接使用$A_{t+1}$去更新我们的Q表格。

Q-learning和Sarsa的区别？
> 1. 首先，Q-learning 是 off-policy 的时序差分学习方法，Sarsa 是 on-policy 的时序差分学习方法。
> 2. 其次，Sarsa 在更新 Q 表格的时候，它用到的$A'$。我要获取下一个 Q 值的时候，$A'$是下一个 step 一定会执行的 action 。这个 action 有可能是$\epsilon-greedy$方法采样出来的值，也有可能是$maxQ$对应的action，也有可能是随机动作。但是就是它实实在在执行了的那个动作。
> 3. 但是Q-learning在更新Q表格的时候，它用到这个的Q值$Q(S',a')$对应的那个 action ，它不一定是下一个 step 会执行的实际的 action，因为你下一个实际会执行的那个 action 可能会探索。Q-learning 默认的 action 不是通过 behavior policy 来选取的，它是默认 $A'$为最优策略选的动作，所以 Q-learning 在学习的时候，不需要传入 $A'$，即$a_{t+1}$的值
> 4. 更新公式的对比（区别只在目标计算这一部分）：
> - Sarsa的公式：$R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$
> - Q-learning的公式：$R_{t+1}+\gamma \max_aQ(S_{t+1},a)$
> Sarsa 实际上都是用自己的策略产生了$(s,a,r,s',a')$这一条轨迹。然后拿着$Q(S_{t+1},A_{t+1})$去更新原本的Q值$Q(S_t,A_t)$。但是Q-learning并不需要指导，我实际上选择了哪一个action，它默认下一个动作就是Q最大的那个动作。所以基于此，Sarsa的action通常会更加保守，胆小，而对应的Q-learning的action会更加莽撞，激进。

On-policy和 off-policy 的区别？ 
> 1. Sarsa 就是一个典型的 on-policy 策略，它只用一个$\pi$，为了兼顾探索和利用，所以它训练的时候会显得有点胆小怕事。它在解决悬崖问题的时候，会尽可能地离悬崖边上远远的，确保说哪怕自己不小心探索了一点了，也还是在安全区域内不不至于跳进悬崖。
> 2. Q-learning 是一个比较典型的 off-policy 的策略，它有目标策略 target policy，一般用$\pi$来表示。然后还有行为策略 behavior policy，用$\mu$来表示。它分离了目标策略跟行为策略。Q-learning就可以大胆地用behavior policy去探索得到的经验轨迹来去优化我的目标策略。这样子我更有可能去探索到最优的策略。
> 3. 比较Q-learning和Sarsa的更新公式可以发现，Sarsa并没有选取最大值的max操作。因此，Q-learning是一个非常激进的算法，希望每一步都获得最大的利益；而Sarsa则相对非常保守，会选择一条相对安全的迭代路线。

### 面试题

能否简述on-policy和off-policy的区别？
> - 答： off-policy和on-policy的根本区别在于生成样本的policy和参数更新时的policy是否相同。**对于on-policy，行为策略和要优化的策略是一个策略，更新了策略后，就用该策略的最新版本对于数据进行采样**；**对于off-policy，使用任意的一个行为策略来对于数据进行采样，并利用其更新目标策略**。如果举例来说，Q-learning在计算下一状态的预期收益时使用了max操作，直接选择最优动作，而当前policy并不一定能选择到最优的action，因此这里生成样本的policy和学习时的policy不同，所以Q-learning为off-policy算法；相对应的SARAS则是基于当前的policy直接执行一次动作选择，然后用这个样本更新当前的policy，因此生成样本的policy和学习时的policy相同，所以SARAS算法为on-policy算法。

能否讲一下Q-Learning，最好可以写出其$Q(s,a)$的更新公式。另外，它是on-policy还是off-policy，为什么？
> - 答： Q-learning是通过计算最优动作值函数来求策略的一种时序差分的学习方法，其更新公式为：$$Q(s,a) \leftarrow Q(s,a) + \alpha[r(s,a)+\gamma \max_{a'}Q(s',a') - Q(s,a)]$$ ，Q-learning是off-policy的，由于是Q更新使用了下一个时刻的最大值，所以我们只关心哪个动作使得$Q(s_{t+1},a)$取得最大值，而实际到底采取了哪个动作（行为策略），并不关心。这表明优化策略并没有用到行为策略的数据，所以说它是off-policy的。

能否讲一下SARSA，最好可以写出其Q(s,a)的更新公式。另外，它是on-policy还是off-policy，为什么？
> - 答：SARSA是on-policy的时序差分算法，它的行为策略和要优化的策略是一个策略，更新了策略后，就用该策略的最新版本对于数据进行采样。其更新公式为：$$Q(s,a) \leftarrow Q(s,a) + \alpha[r(s,a)+\gamma Q(s',a') - Q(s,a)]$$。Sarsa是on-policy的，Sarsa必须执行两次动作得到$(s,a,r,s',a')$才可以更新一次；而$a'$是在特定策略$\pi$的指导下执行的动作，因此估计出来的$Q(s,a)$是在该策略$\pi$之下的Q-value，样本生成用的$\pi$和估计的$\pi$是同一个，因此是on-policy。

请问value-based和policy-based的区别是什么？
> 1. **生成policy上的差异：前者确定，后者随机**。Value-Base中的action-value估计值最终会收敛到对应的true values（通常是不同的有限数，可以转化为0到1之间的概率），因此通常会获得一个确定的策略（deterministic policy）；而Policy-Based不会收敛到一个确定性的值，另外他们会趋向于生成optimal stochastic policy。如果optimal policy是deterministic的，那么optimal action对应的性能函数将远大于suboptimal actions对应的性能函数，性能函数的大小代表了概率的大小。
> 2. **动作空间是否连续，前者离散，后者连续**。Value-Base，对于连续动作空间问题，虽然可以将动作空间离散化处理，但离散间距的选取不易确定。过大的离散间距会导致算法取不到最优action，会在这附近徘徊，过小的离散间距会使得action的维度增大，会和高维度动作空间一样导致维度灾难，影响算法的速度；而Policy-Based适用于连续的动作空间，在连续的动作空间中，可以不用计算每个动作的概率，而是通过Gaussian distribution （正态分布）选择action。
> 3. value-based，例如Q-learning，是通过求解最优值函数间接的求解最优策略；policy-based，例如REINFORCE，Monte-Carlo Policy Gradient，等方法直接将策略参数化，通过策略搜索，策略梯度或者进化方法来更新策略的参数以最大化回报。基于值函数的方法不易扩展到连续动作空间，并且当同时采用非线性近似、自举和离策略时会有收敛性问题。策略梯度具有良好的收敛性证明。
> 4. 补充：对于值迭代和策略迭代：**策略迭代有两个循环**，一个是在策略估计的时候，为了求当前策略的值函数需要迭代很多次。另外一个是外面的大循环，就是策略评估，策略提升这个循环。**值迭代算法则是一步到位，直接估计最优值函数，因此没有策略提升环节**。

请简述以下时序差分(Temporal Difference，TD)算法。
> - 答：TD算法是使用广义策略迭代来更新Q函数的方法，核心使用了自举（bootstrapping），即值函数的更新使用了下一个状态的值函数来估计当前状态的值。也就是使用下一步的$Q$值$Q(S_{t+1},A_{t+1})$来更新我这一步的Q值$Q(S_t,A_t)$。完整的计算公式如下：$$Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha[R_{t+1},\gamma Q(S_{t+1},A_{t+1})]$$

问蒙特卡洛方法（Monte Carlo Algorithm，MC）和时序差分(Temporal Difference，TD)算法是无偏估计吗？另外谁的方法更大呢？为什么呢？
> - 答：**蒙特卡洛方法（MC）是无偏估计**，**时序差分（TD）是有偏估计**；MC的方差较大，TD的方差较小，原因在于TD中使用了自举（bootstrapping）的方法，实现了基于平滑的效果，导致估计的值函数的方差更小。

能否简单说下动态规划、蒙特卡洛和时序差分的异同点？
> - 相同点：都用于进行值函数的描述与更新，并且所有方法都是基于对未来事件的展望来计算一个回溯值。
> - 不同点：蒙特卡洛和TD算法隶属于model-free，而动态规划属于model-based；TD算法和蒙特卡洛的方法，因为都是基于model-free的方法，因而对于后续状态的获知也都是基于试验的方法；TD算法和动态规划的策略评估，都能基于当前状态的下一步预测情况来得到对于当前状态的值函数的更新。
> - 另外，TD算法不需要等到实验结束后才能进行当前状态的值函数的计算与更新，而蒙特卡洛的方法需要试验交互，产生一整条的马尔科夫链并直到最终状态才能进行更新。TD算法和动态规划的策略评估不同之处为model-free和model-based 这一点，动态规划可以凭借已知转移概率就能推断出来后续的状态情况，而TD只能借助试验才能知道。
> - 蒙特卡洛方法和TD方法的不同在于，蒙特卡洛方法进行完整的采样来获取了长期的回报值，因而在价值估计上会有着更小的偏差，但是也正因为收集了完整的信息，所以价值的方差会更大，原因在于毕竟基于试验的采样得到，和真实的分布还是有差距，不充足的交互导致的较大方差。而TD算法与其相反，因为只考虑了前一步的回报值 其他都是基于之前的估计值，因而相对来说，其估计值具有偏差大方差小的特点。
> - 三者的联系：对于$TD(\lambda)$方法，如果$\lambda=0$，那么此时等价于$TD$，即只考虑下一个状态；如果$\lambda=1$，等价于$MC$，即考虑$T-1$个后续状态，即到整个episode序列结束。

### 实验

#### Q-learning 实验结果

- 第一次训练结果
<img src="https://docs-xy.oss-cn-shanghai.aliyuncs.com/rl02.png" alt="ql01" style="width:60%">


- 读取了上一次训练结果（策略）之后的第二次训练
<img src="https://docs-xy.oss-cn-shanghai.aliyuncs.com/rl03.png" alt="ql02" style="width:60%">

#### Sarsa 实验结果

- 第一次训练结果
<img src="https://docs-xy.oss-cn-shanghai.aliyuncs.com/rl04.png" alt="Sarsa01" style="width:60%">

- 读取了上一次训练结果（策略）之后的第二次训练
<img src="https://docs-xy.oss-cn-shanghai.aliyuncs.com/rl05.png" alt="Sarsa02" style="width:60%">

#### 实验结果分析

在无法获取马尔可夫决策过程的模型情况下，我们可以通过蒙特卡洛方法和时序差分方法来估计某个给定策略的价值。重点关注表格型策略中的Q-learning，Sarsa方法，它们都是免模型的强化学习方法。区别在于Q-learning是off-policy的，Sarsa是on-policy的。

Q-learning分离了目标策略跟行为策略。大胆地用行为策略去探索得到的经验轨迹来去优化我的目标策略。Q-learning为了获得最优策略，在选择Q值最大的动作的同时，这一action还可能进行探索，导致最终reward的波动较大，通俗地说Q-learning是激进的，但Q-learning可以找到最优策略，悬崖寻路问题中可以找到理论最大reward（-13）。

Sarsa只使用了一个特定策略，Sarsa并没有选取最大值的max操作，Sarsa则是基于当前的policy直接执行一次动作选择，然后用这个样本更新当前的policy，因此生成样本的policy和学习时的policy相同。Sarsa通俗来讲是保守的，它在悬崖寻路问题中找到的最大reward是-15。

## 第四章 策略梯度
https://datawhalechina.github.io/easy-rl/#/chapter4/chapter4

## 第五章 PPO
https://datawhalechina.github.io/easy-rl/#/chapter5/chapter5

## 第六章 DQN 基本概念

DQN创新点：
1.  经验回放解决了两个问题：
    1. 序列决策的样本关联 
    2. 样本利用率低；
2. 固定Q目标解决算法非平稳性的问题。

经验回放充分利用了off-policy的优势，DQN实现了含有$\epsilon-greedy$的Sample函数保证所有动作能被探索到，实现了Learning方法使智能体与环境交互的数据能够交付给模型。

智能体与环境交互得到的数据存入Replay Buffer，从经验池中Sample一个batch的数据送给learn函数。Q网络用于产生Q预测值，target Q用于产生Q目标值，定期从Q网络复制参数至target Q网络，Q预测值与Q目标值作为Loss，根据Loss再去更新Q网络。

<img src="https://docs-xy.oss-cn-shanghai.aliyuncs.com/rl06.png" alt="dqn" style="width:60%">

