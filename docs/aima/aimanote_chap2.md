## 第二章

- **智能体**是在环境中感知和行动的事物。智能体的**智能体函数**指定智能体在响应任何感知序列时所采取的动作。
- **性能度量**评估智能体在环境中的行为。给定到目前为止所看到的感知序列时，**理性智能体**的动作是为了最大化性能度量的期望值。
- **任务环境**规范包括性能度量、外部环境、执行器和传感器。在设计智能体时，第一步必须始终是尽可能地指定任务环境。
- 任务环境在几个重要维度上有所不同。它们可以是完全可观测的或部分可观测的、单智能体的或多智能体的、确定性的或非确定性的、回合式的或序贯的、静态的或动态的、离散的或连续的、已知的或未知的。
- 在性能度量未知或难以正确指定的情况下，智能体优化错误目标的风险很大。在这种情况下，智能体设计应该反映真实目标的不确定性。
- **智能体程序**实现智能体函数。存在各种基本的智能体编程，反映了决策过程中明确使用的信息类型。这些设计在效率、紧凑性和灵活性方面各不相同。智能体程序的适当设计取决于环境的性质。
- **简单反射型智能体**直接响应感知，而**基于模型的反射型智能体**保持内部状态以跟踪当前感知中不明晰的世界状态。**基于目标的智能体**采取行动来实现目标，而**基于效用的智能体**试图最大化自己期望的“快乐”。
- 所有智能体都可以通过**学习**提升性能。


## 第二章 习题与答案
> Part I Artificial Intelligence: 2. Intelligent Agents

### Exercise 1

:question:  **Question 1:**

假设性能度量只关注环境的前T个时间步长，并忽略此后的所有内容。表明理性智能体的行为可能不仅取决于环境的状态，还取决于它所达到的时间步长。
> Suppose that the performance measure is concerned with just the first T time steps of the environment and ignores everything thereafter. Show that a rational agent’s action may depend not just on the state of the environment but also on the time step it has reached.

:exclamation: **Answer 1:**

例如：一个交互式的智能英语教师，它的教学行为不仅取决于学生当前的英语水平，还取决于它已经教了多少课程。如果它的性能度量只关注前T个时间步长，那么它可能会在T个时间步长后停止教学，即使它的学生还没有达到它的期望水平。区别于后T个时间步长，前T个时间步长的性能度量是可知的，因此可以通过学习来优化；类比于已经教过的课程，智能英语教师可以不断改良课程，使得学生的英语水平达到它的期望水平。

---

### Exercise 2

:question:  **Question 2:**

让我们来看看各种吸尘器智能体函数的合理性。
<img src="https://docs-xy.oss-cn-shanghai.aliyuncs.com/aima2-3.png" alt="2-3" style="width:85%">

1. 表明图 2.3 中描述的简单吸尘器智能体函数在页面中列出的假设下确实是合理的
2. 描述一个理性智能体函数，说明每次运动成本一个点的情况。相应的智能体程序是否需要内部状态？
3. 讨论清洁方块可能变脏且环境地理未知的可能智能体设计。在这些情况下，智能体从其经验中学习是否有意义？如果是这样，它应该学习什么？如果没有，为什么不呢？
> Let us examine the rationality of various vacuum-cleaner agent functions.
> 1. Show that the simple vacuum-cleaner agent function described in Figure 2.3 is indeed rational under the assumptions listed on page 
> 2. Describe a rational agent function for the case in which each movement costs one point. Does the corresponding agent program require internal state?
> 3. Discuss possible agent designs for the cases in which clean squares can become dirty and the geography of the environment is unknown. Does it make sense for the agent to learn from its experience in these cases? If so, what should it learn? If not, why not?

:exclamation: **Answer 2:**

1. 一个只有两个方格的吸尘器世界，当A格干净时，向右移动到B格，当B格干净时，向左移动到A格。当所处的格子脏时，进行吸尘。这个智能体函数是合理的，因为它能够保证每个格子都被清洁，而且不会重复清洁。这个智能体函数的性能度量是：每个格子被清洁一次消耗一个点，每次移动消耗一个点。这个智能体函数的性能度量是可知的，因此可以通过学习来优化。
2. 首先，智能体程序和智能体函数不同，智能体程序将当前的感知作为输入，智能体函数可能以来整个感知历史，需要将两个概念区分开来。对于智能体程序，它需要内部状态，因为它需要记录当前所处的格子，以便在下一次感知时，知道它应该向哪个方向移动。对于智能体函数，它不需要内部状态，因为它只需要知道当前所处的格子，而不需要知道它之前所处的格子。
3. 当方块可能变脏时，从经验中学习时有意义的。因为它可以从经验中学习哪些方块可能会在什么时候有多大的概率变脏。它应该学习的是，当它清洁了一个方块时，它应该记录下这个方块的清洁时间，当它再次清洁这个方块时，如果这个方块的清洁时间距离上次清洁时间很近，那么它应该认为这个方块很可能会变脏。但是，当环境地理未知时，从经验中学习是没有意义的。因为它不知道它所处的格子是哪个格子，所以它无法从经验中学习哪些方块可能会在什么时候有多大的概率变脏。

---

### Exercise 3
:question: **Question 3:**

写一篇关于进化与自主、智力和学习的一种或多种关系的文章。
> Write an essay on the relationship between evolution and one or more of autonomy, intelligence, and learning.
:exclamation: **Answer 3:**

进化、自主、智力和学习之间有着复杂的关系。进化是一种长期的过程，它影响着生物体的自主性和智力。自主性和智力又会影响学习能力。

进化使得生物体获得了更高的智力和自主性。例如，人类的大脑是非常复杂的，它具有许多不同的区域，可以处理各种不同的信息。这使得人类能够进行更复杂的思考和决策。自主性也与进化有关，随着生物体演化出更高的自主性，它们能够更好地掌控自己的行为和决策。

智力和自主性又会影响学习能力。生物体具有较高的智力和自主性，可以更好地理解和学习新信息。人类是一个很好的例子，因为我们具有高度的智力和自主性，我们能够进行复杂的学习和思考。

总之，进化、自主性、智力和学习之间有着密切的关系。进化导致生物体具有更高的智力和自主性，而这又会影响学习能力。

---

### Exercise 4
:question: **Question 4:**

对于以下每个断言，请说出它是对还是错，并在适当的情况下用示例或反例来支持您的答案。
1. 一个只感知到关于状态的部分信息的主体不可能是完全理性的。
2. 存在任务环境，在这种环境中，没有纯粹的反射智能体能够理性地工作。
3. 存在一个任务环境，其中每个智能体都是理性的。
4. 智能体程序的输入与智能体函数的输入相同。
5. 每个智能体函数都可以通过一些程序/机器组合来实现。
6. 假设智能体从可能的一组操作中随机统一选择其操作。存在一个确定性任务环境，在该环境中，此智能体是合理的。
7. 给定的智能体在两个不同的任务环境中可能是完全理性的。
8. 在不可观察的环境中，每个智能体都是理性的。
9. 一个完全理性的扑克玩家永远不会输。
> For each of the following assertions, say whether it is true or false and support your answer with examples or counterexamples where appropriate.
> 1. An agent that senses only partial information about the state cannot be perfectly rational.
> 2. There exist task environments in which no pure reflex agent can behave rationally.
> 3. There exists a task environment in which every agent is rational.
> 4. The input to an agent program is the same as the input to the agent function.
> 5. Every agent function is implementable by some program/machine combination.
> 6. Suppose an agent selects its action uniformly at random from the set of possible actions. There exists a deterministic task environment in which this agent is rational.
> 7. It is possible for a given agent to be perfectly rational in two distinct task environments.
> 8. Every agent is rational in an unobservable environment.
> 9. A perfectly rational poker-playing agent never loses.

:exclamation: **Answer 4:**

1. 这是对的。例如，如果一个人只知道有关某个投票的部分信息，那么他就不能做出完全理性的决策。在这种情况下，该人可能会做出基于偏见或不完整信息的决策。
2. 这是对的。纯反射智能体只会对刺激作出反应，而不会考虑上下文或长期目标。因此，在任何需要考虑上下文或长期目标的任务环境中，纯反射智能体都不能合理地行事。例如，在游戏中，如果机器人只是对每个按钮的按钮做出反应，而不是根据当前游戏状态和目标来决定哪些按钮需要按下，那么它将不能赢得游戏。
3. 这是错的。理性是一个相对的概念，因此不可能存在一个任务环境，其中每个智能体都是完全理性的。有时，一个智能体可能因为缺乏信息或受到偏见的影响而做出不理性的决策。此外，在某些情况下，一个智能体可能需要在理性和非理性之间权衡，因此不可能是完全理性的。
4. 这是错的。智能体程序是一组指令，用于控制智能体行为的计算机程序。而智能体函数是一个算法，用于控制智能体行为。因此，智能体程序的输入和智能体函数的输入可能不同。智能体程序可能需要收集和处理传感器数据，以便将其作为智能体函数的输入。
5. 这是对的。智能体函数是一种算法，可以通过编写程序或使用已有的机器来实现。例如，在机器学习中，智能体函数可以通过训练模型来实现，而在规划中，智能体函数可以通过编写规划程序来实现。因此，每个智能体函数都可以通过编程和机器组合来实现。
6. 这是错的。确定性任务环境是指环境中的状态和结果是确定的。而随机选择一组操作显然是不合理的，因为它不考虑环境中状态的影响，也不考虑长期目标。因此，在确定性任务环境中，这样的智能体不能做出合理的决策。
7. 这是对的。理性是相对的概念，它取决于环境和目标。在不同的任务环境中，智能体可能会使用不同的策略来实现其目标。例如，在一个任务环境中，智能体可能会优先考虑节约能量，而在另一个环境中，它可能会优先考虑快速完成任务。这些策略都可能是合理的，因此在不同的任务环境中，智能体可能是完全理性的。
8. 这是错的。在不可观察的环境中，智能体可能无法获取足够的信息来做出理性的决策。例如，在一个没有任何可观测信息的环境中，智能体可能无法确定它所处的位置，这将导致它做出错误的决策。另外，智能体可能受到数据偏差或错误设计等因素的影响而导致不理性.
9. 这是错的。虽然完全理性的扑克玩家可能会使用最佳策略来赢得游戏，但是这并不意味着他永远不会输。因为扑克是一种随机游戏，即使使用最佳策略，也有可能会在某些局面下输。此外,其他玩家也可能会使用最优策略，所以完全理性的扑克玩家也有可能在某些情况下输给其他玩家。

--- 
### Exercise 5
:question: **Question 5:**

对于以下每个活动，请对任务环境进行 PEAS 描述，并根据部分中列出的属性对其进行表征
- 踢足球。
- 探索土卫六的地下海洋。
- 在互联网上购买二手AI书籍。
- 打网球比赛。
- 靠墙练习网球。
- 跳高。
- 编织毛衣。
- 在拍卖会上竞标物品。
> For each of the following activities, give a PEAS description of the task environment and characterize it in terms of the properties listed in Section 
>- Playing soccer.
>- Exploring the subsurface oceans of Titan.
>- Shopping for used AI books on the Internet.
>- Playing a tennis match.
>- Practicing tennis against a wall.
>- Performing a high jump.
>- Knitting a sweater.
>- Bidding on an item at an auction.

:exclamation: **Answer 5:**

**PEAS:**

|任务环境|Performance measure|Environment|Actuators|Sensors|
|-|-|-|-|-|
|踢足球|进球数|足球场上|脚|眼睛和耳朵|
|探索土卫六的地下海洋|收集到的科学数据和样本|土卫六的地下海洋|机器人或潜水艇|摄像机和传感器|
|在互联网上购买二手AI书籍|购买的书籍的质量和数量|互联网|鼠标和键盘|眼睛|
|打网球比赛|赢得的局数|网球场|手和身体|眼睛和耳朵|
|靠墙练习网球|击球精度和速度|网球场边墙|手和身体|眼睛和耳朵|
|跳高|跳过的高度|跳高场|脚|眼睛和耳朵|
|编织毛衣|毛衣的质量和外观|工作室或家|手|眼睛和手|
|在拍卖会上竞标物品|获得的物品的价值|拍卖会|鼠标和键盘|眼睛|

**属性：**
| 任务环境 | 可观测     | 智能体 | 确定性 | 回合式 | 静态   | 离散 |
| :------------------------: | ---------- | :----: | :----: | :----: | ------ | ---- |
|           踢足球           | 部分可观测 |   多   | 不确定 |  回合  | 动态   | 连续 |
|    探索土卫六的地下海洋    | 部分可观测 |   单   | 不确定 |  序贯  | 半动态 | 连续 |
| 在互联网上购买二手 AI 书籍 | 完全可观测 |   单   |  确定  |  序贯  | 静态   | 离散 |
|         打网球比赛         | 部分可观测 |   多   | 不确定 |  序贯  | 动态   | 连续 |
|        靠墙练习网球        | 完全可观测 |   单   |  确定  |  回合  | 动态   | 连续 |
|          表演跳高          | 完全可观测 |   单   |  确定  |  回合  | 静态   | 离散 |
|           织毛衣           | 完全可观测 |   单   |  确定  |  序贯  | 动态   | 连续 |
|   在拍卖会上竞标一件物品   | 部分可观测 |   多   | 不确定 |  序贯  | 动态   | 连续 |

---

### Exercise 6
:question: **Question 6:**

Same to Question 5
> Same to Question 5

:exclamation: **Answer 6:**

Same to Answer 5

---

### Exercise 7
:question: **Question 7:**

用你自己的话定义以下术语：智能体，智能体函数，智能体程序，理性，自主性，反射智能体，基于模型的智能体，基于目标的智能体，基于效用的智能体，学习智能体。
> Define in your own words the following terms: agent, agent function, agent program, rationality, autonomy, reflex agent, model-based agent, goal-based agent, utility-based agent, learning agent.

:exclamation: **Answer 7:**

- 智能体: 智能体是一种能够感知环境，做出决策并采取行动的系统。
- 智能体函数: 智能体函数是一种算法，用来描述智能体如何根据感知到的环境信息来做出决策。
- 智能体程序: 智能体程序是一组指令，用来控制智能体如何感知环境，做出决策并采取行动。
- 理性: 理性是指一种决策方式，在给定的环境和目标的情况下，能够使用最优策略来实现目标。
- 自主性: 自主性是指智能体能够自己决定和执行行动的能力。
- 反射智能体: 反射智能体是一种基于规则的智能体，它根据感知到的环境信息采取行动，而不是基于策略或目标。
- 基于模型的智能体: 基于模型的智能体是一种使用模型来预测环境行为和结果的智能体。
- 基于目标的智能体: 基于目标的智能体是一种根据目标来决定行动的智能体。
- 基于效用的智能体: 基于效用的智能体是一种根据效用函数来评估环境状态

---

### Exercise 8 
:question: **Question 8:**

本练习探讨智能体函数和智能体程序之间的差异。
1. 是否可以有多个智能体程序实现给定的智能体功能？举个例子，或者说明为什么不可能。
2. 是否存在任何智能体程序都无法实现的智能体函数？
3. 给定固定的机器架构，每个智能体程序是否只实现一个智能体函数？
4. 给定一个具有 n 位存储的架构，有多少种不同的可能智能体程序？
5. 假设我们保持智能体程序固定，但将机器速度提高两倍。这会改变智能体函数吗？
> This exercise explores the differences between agent functions and agent programs.
> 1. Can there be more than one agent program that implements a given agent function? Give an example, or show why one is not possible.
> 2. Are there agent functions that cannot be implemented by any agent program?
> 3. Given a fixed machine architecture, does each agent program implement exactly one agent function?
> 4. Given an architecture with n bits of storage, how many different possible agent programs are there?
> 5. Suppose we keep the agent program fixed but speed up the machine by a factor of two. Does that change the agent function?

:exclamation: **Answer 8:**

1. 可能有多种智能体程序实现给定的智能体功能，因为不同的程序可能采用不同的算法和技术来达到相同的目标。例如，两个不同的程序可能都能实现一个自动驾驶汽车，但它们可能采用不同的感知、规划和控制技术。
2. 可能存在某些智能体函数，其不能由任何现有的智能体程序实现，因为当前技术和算法可能还无法解决某些问题。
3. 不一定，一个智能体程序可能实现多个智能体函数，或者一个智能体函数可能由多个智能体程序来实现。
4. 由于存储空间的限制，不同的智能体程序的数量可能是有限的。具体数量取决于每个程序的大小和存储空间的容量。
5. 不会，提高机器速度只会影响程序的运行速度，而不会改变程序的功能。

### Exercise 9
:question: **Question 9:**

为基于目标和基于效用的智能体编写伪代码智能体程序。
> Write pseudocode agent programs for the goal-based and utility-based agents.

:exclamation: **Answer 9:**

- 基于目标的智能体程序伪代码
```
1. 设置目标 goal
2. 重复执行步骤3至5，直到目标被达成
3. 选择当前状态下可能的下一个状态
4. 根据目标选择最优的下一个状态
5. 执行动作并更新当前状态
6. 目标已达成
```
- 基于效用的智能体程序伪代码

```
1. 设置效用函数 utility function
2. 重复执行步骤3至5，直到没有更优的状态
3. 选择当前状态下可能的下一个状态
4. 计算每个下一个状态的效用值
5. 选择效用值最高的下一个状态
6. 执行动作并更新当前状态
7. 没有更优的状态
```

---
> 以下练习都涉及吸尘器世界的环境和智能体的实现。
> 
> The following exercises all concern the implementation of environments and agents for the vacuum-cleaner world.

--- 
### Exercise 10

:question: **Question 10:**

考虑一个简单的恒温器，当温度至少低于设置3度时打开炉子，当温度至少高于设置3度时关闭炉子。恒温器是简单反射智能体、基于模型的反射智能体还是基于目标的智能体的实例？
> Consider a simple thermostat that turns on a furnace when the temperature is at least 3 degrees below the setting, and turns off a furnace when the temperature is at least 3 degrees above the setting. Is a thermostat an instance of a simple reflex agent, a model-based reflex agent, or a goal-based agent?

:exclamation: **Answer 10:**

恒温器是一个简单反射智能体的实例，它在特定的环境下执行特定的动作，而不是根据模型或长期目标来进行决策的。

---

### Exercise 11
:question: **Question 11:**

为图 2.8 中描述并在页面中指定的吸尘器世界实现性能测量环境模拟器。您的实现应该是模块化的，以便可以轻松更改传感器、执行器和环境特征（尺寸、形状、污垢放置等）。（注意：对于某些编程语言和操作系统的选择，在线代码存储库中已经有实现）
> Implement a performance-measuring environment simulator for the vacuum-cleaner world depicted in Figure 2.8 and specified on page . Your implementation should be modular so that the sensors, actuators, and environment characteristics (size, shape, dirt placement, etc.) can be changed easily. (Note: for some choices of programming language and operating system there are already implementations in the online code repository.)

<img src="https://docs-xy.oss-cn-shanghai.aliyuncs.com/aima2-8.png" alt="2-8" style="width:100%">

:exclamation: **Answer 11:**

```python
def ReflexVacuumAgent():
    """
    [Figure 2.8]
    A reflex agent for the two-state vacuum environment.
    >>> agent = ReflexVacuumAgent()
    >>> environment = TrivialVacuumEnvironment()
    >>> environment.add_thing(agent)
    >>> environment.run()
    >>> environment.status == {(1,0):'Clean' , (0,0) : 'Clean'}
    True
    """

    def program(percept):
        location, status = percept
        if status == 'Dirty':
            return 'Suck'
        elif location == loc_A:
            return 'Right'
        elif location == loc_B:
            return 'Left'

    return Agent(program)
```

---

### Exercise 12
:question: **Question 12:**

为吸尘器环境实现一个简单反射智能体。运行这个智能体的环境，包含所有可能的初始污垢位置和智能体位置。记录每个配置的性能得分和总体平均得分。
> Implement a simple reflex agent for the vacuum environment in Exercise vacuum-start-exercise. Run the environment with this agent for all possible initial dirt configurations and agent locations. Record the performance score for each configuration and the overall average score.

:exclamation: **Answer 12:**

代码示例来源：https://github.com/aimacode/aima-python/blob/master/agents.py

```python
class TrivialVacuumEnvironment(Environment):
    """This environment has two locations, A and B. Each can be Dirty
    or Clean. The agent perceives its location and the location's
    status. This serves as an example of how to implement a simple
    Environment."""

    def __init__(self):
        super().__init__()
        self.status = {loc_A: random.choice(['Clean', 'Dirty']),
                       loc_B: random.choice(['Clean', 'Dirty'])}

    def thing_classes(self):
        return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent, TableDrivenVacuumAgent, ModelBasedVacuumAgent]

    def percept(self, agent):
        """Returns the agent's location, and the location status (Dirty/Clean)."""
        return agent.location, self.status[agent.location]

    def execute_action(self, agent, action):
        """Change agent's location and/or location's status; track performance.
        Score 10 for each dirt cleaned; -1 for each move."""
        if action == 'Right':
            agent.location = loc_B
            agent.performance -= 1
        elif action == 'Left':
            agent.location = loc_A
            agent.performance -= 1
        elif action == 'Suck':
            if self.status[agent.location] == 'Dirty':
                agent.performance += 10
            self.status[agent.location] = 'Clean'

    def default_location(self, thing):
        """Agents start in either location at random."""
        return random.choice([loc_A, loc_B])
```

---

### Exercise 13

:question: **Question 13:**

考虑练习 2.10 中吸尘器环境的修改版本，其中智能体每移动一个动作就会被惩罚一分。
1. 对于这种环境，一个简单的反射智能体可以完全合理吗？解释。
2. 有状态的反射智能体呢？设计这样的智能体。
3. 如果智能体的感知赋予环境中每个方块的干净/肮脏状态，您对 1 和 2 的答案会如何变化？
> Consider a modified version of the vacuum environment in Exercise 2.10, in which the agent is penalized one point for each movement.
> 1. Can a simple reflex agent be perfectly rational for this environment? Explain.
> 2. What about a reflex agent with state? Design such an agent.
> 3. How do your answers to 1 and 2 change if the agent’s percepts give it the clean/dirty status of every square in the environment?

:exclamation: **Answer 13:**

1. 不完全合理，因为它只能根据当前状态执行动作，这可能会导致智能体陷入无限循环或选择不优秀的策略。
2. 有状态的反射智能体可以考虑环境中的惩罚因素。例如，智能体可以在移动之前先扫描周围的区域，确定哪些区域需要清洁，并确定最优策略。
3. 如果智能体的感知能够提供环境中每个方块的干净/肮脏状态，这会使 1 和 2 的答案变得更合理。智能体可以根据干净/肮脏状态来优化其策略，并在不需要清洁的区域中省略移动操作。

---

### Exercise 14

:question: **Question 14:**

考虑练习 2.10 中吸尘器环境的修改版本，其中环境的地理范围（范围、边界和障碍物）是未知的，初始污垢配置也是如此。（代理可以上下以及左右移动。
1. 对于这种环境，一个简单的反射智能体可以完全合理吗？解释。
2. 具有随机智能体函数的简单反射智能体能否优于单纯反射智能体？设计此类智能体并测量其在多个环境中的性能。
3. 你能设计一个随机智能体表现不佳的环境吗？显示您的结果。
4. 有状态的智能体能胜过单纯的智能体吗？设计此类智能体并测量其在多个环境中的性能。你能设计出这种类型的理性智能体吗？
> Consider a modified version of the vacuum environment in Exercise 2.10, in which the geography of the environment—its extent, boundaries, and obstacles—is unknown, as is the initial dirt configuration. (The agent can go Up and Down as well as Left and Right.)
> 1. Can a simple reflex agent be perfectly rational for this environment? Explain.
> 2. Can a simple reflex agent with a randomized agent function outperform a simple reflex agent? Design such an agent and measure its performance on several environments.
> 3. Can you design an environment in which your randomized agent will perform poorly? Show your results.
> 4. Can a reflex agent with state outperform a simple reflex agent? Design such an agent and measure its performance on several environments. Can you design a rational agent of this type?

:exclamation: **Answer 14:**

1. 一个简单的反射智能体在这种环境中不能完全合理，因为地理范围和初始污垢配置是未知的。智能体不能根据这些信息来优化其策略，因此可能会陷入无限循环或者不能清洁整个环境
2. 具有随机智能体函数的简单反射智能体可能会优于单纯反射智能体。这种智能体可以在不知道地理范围和初始污垢配置的情况下，通过随机移动来探索环境，并且在发现污垢时执行清洁动作。为了测量其在多个环境中的性能，需要进行大量的实验并统计数据。
3. 可以通过设计一个环境，其中障碍物和污垢都很密集，并且环境中没有明显的规律来让随机智能体表现不佳。在这样的环境中，随机智能体可能会花费大量时间来探索环境，而不能有效地清洁环境。
4. 有状态的智能体可能会胜过单纯的智能体。这种智能体可以根据之前的经验来优化其策略并使用最优策略来清洁环境。为了设计这样的智能体并测量其在多个环境中的性能，可以使用机器学习算法，如 Q-learning 或 SARSA。这样的智能体可以通过不断学习和改进来达到理性的行为。

---

### Exercise 15

:question: **Question 15:**

对于将位置传感器替换为“碰撞”传感器的情况，重复练习 2.13，该传感器检测智能体试图进入障碍物或越过环境边界的情况。假设碰撞传感器停止工作；智能体应该如何表现？
> Repeat Exercise 2.13 for the case in which the location sensor is replaced with a “bump” sensor that detects the agent’s attempts to move into an obstacle or to cross the boundaries of the environment. Suppose the bump sensor stops working; how should the agent behave?

:exclamation: **Answer 15:**

如果碰撞传感器停止工作，智能体将无法感知到它试图进入障碍物或越过环境边界的情况。这可能会导致智能体在环境中陷入死循环或者不能清洁整个环境，因为它不能知道自己在哪里和它试图移动的方向是否正确。

在这种情况下，如果智能体不能感知到它试图进入障碍物或越过环境边界的情况，那么它将无法避免这种情况，并可能越出环境、在环境中陷入死循环或者不能清洁整个环境。

---

### Exercise 16
:question: **Question 16:**

前面练习中的传感器环境都是确定性的。讨论以下每个随机版本的可能智能体程序：
1.墨菲定律：百分之二十五的情况下，如果地板脏了，吸尘动作无法清洁地板，如果地板干净，则灰尘沉积在地板上。如果污垢传感器在 10% 的时间内给出错误的答案，您的智能体程序会受到怎样的影响？
2.小孩子：在每个时间步长，每个干净的方块都有10%的几率变脏。你能为这个案例想出一个合理的智能体设计吗？
> The vacuum environments in the preceding exercises have all been deterministic. Discuss possible agent programs for each of the following stochastic versions:
> 1. Murphy’s law: twenty-five percent of the time, the Suck action fails to clean the floor if it is dirty and deposits dirt onto the floor if the floor is clean. How is your agent program affected if the dirt sensor gives the wrong answer 10% of the time?
> 2. Small children: At each time step, each clean square has a 10% chance of becoming dirty. Can you come up with a rational agent design for this case?

:exclamation: **Answer 16:**

1. 如果污垢传感器在 10% 的时间内给出错误的答案,在这种情况下，程序可能会在地板干净的情况下进行吸尘动作。或者，程序可能会在地板脏的情况下不进行吸尘动作，导致地板不能得到清洁。为了解决这个问题，可能需要引入额外的传感器来确认地板的清洁状态，或者采用一种更加灵活的策略来应对随机性。
2. 对于这个随机版本的案例，可以考虑使用一种基于频率的清洁策略。例如，智能体程序可以在一定的时间间隔内对整个环境进行清洁，而不是等待地板变脏再进行清洁。这样可以最大限度地降低地板变脏的可能性，从而提高清洁效率。


