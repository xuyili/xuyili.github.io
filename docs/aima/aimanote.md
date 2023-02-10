# 人工智能：现代方法 第四版

## 第一章

- 不同的人对人工智能的期望不同。首先要问的两个重要问题是：你关心的是思想还是行为？你想模拟人类，还是试图达到最佳结果？
- 根据我们所说的标准模型，人工智能主要关注理性行为。理想的智能体会在某种情况下采取可能的最佳行为，在这个意义下，我们研究了智能体的构建问题。
- 这个简单的想法需要两个改进：首先，任何智能体（无论是人还是其他物体）选择理性行为的能力都受到决策计算难度的限制；其次，机器的概念需要从追求明确目标转变到追求目标以造福人类，虽然不确定这些目标是什么。
- 哲学家们（追溯到公元前 400 年）暗示大脑在某些方面就像一台机器，操作用某种内部语言编码的知识，并且这种思维可以用来选择要采取的行动，从而认为人工智能是有可能实现的。
- 数学家提供了运算逻辑的确定性陈述以及不确定的概率陈述的工具，也为理解计算和算法推理奠定了基础。
- 经济学家将决策问题形式化，使决策者的期望效用最大化。
- 神经科学家发现了一些关于大脑如何工作的事实，以及大脑与计算机的相似和不同之处。
- 心理学家采纳了人类和动物可以被视为信息处理机器的观点。语言学家指出，语言的使用符合这一模式。
- 计算机工程师提供了更加强大的机器，使人工智能应用成为可能，而软件工程师使它们更加易用。
- 控制理论涉及在环境反馈的基础上设计最优行为的设备。最初，控制理论的数学工具与人工智能中使用的大不相同，但这两个领域越来越接近。
- 人工智能的历史经历了成功、盲目乐观以及由此导致的热情丧失和资金削减的循环，也存在引入全新创造性的方法和系统地改进最佳方法的循环。
- 与最初的几十年相比，人工智能在理论和方法上都已经相当成熟。随着人工智能面对的问题变得越来越复杂，该领域从布尔逻辑转向概率推理，从手工编码知识转向基于数据的机器学习。这推动了真实系统功能的改进以及与其他学科更大程度的集成。
- 随着人工智能系统在真实世界中的应用，必须考虑各种风险和道德后果。
- 从长远来看，我们面临着控制超级智能的人工智能系统的难题，它们可能以不可预测的方式进化。解决这个问题似乎需要改变我们对人工智能的设想。



## 第一章 习题与答案

> Part I Artificial Intelligence: 1. Introduction
### Exercise 1
:question: **Question 1:** 

用您自己的话来定义：（a）智能，（b）人工智能，（c）智能体，（d）理性，（e）逻辑推理。
> Define in your own words: (a) intelligence, (b) artificial intelligence, (c) agent, (d) rationality, (e) logical reasoning.

:exclamation: **Answer 1:**

（a）智能：是智力和能力的总称，中国古代思想家一般把智与能看做是两个相对独立的概念。也有不少思想家把二者结合起来作为一个整体看待。

（b）人工智能：是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。它研究的一个主要目标是使机器能够胜任一些通常需要人类智能才能完成的复杂工作。

（c）智能体：具有智能的实体，以云为基础，以AI为核心，构建一个立体感知、全域协同、精准判断、持续进化、开放的智能系统。或者简单说能够采取行动，能够自主运行、感知环境、长期持续存在、适应变化以及制定和实现目标。

（d）理性：指人在正常思维状态下时为了获得预期结果，有自信与勇气冷静地面对现状，并快速全面了解现实分析出多种可行性方案，再判断出最佳方案且对其有效执行的能力。理性是基于现有的理论，通过合理的逻辑推导得到确定的结果，

（e）逻辑推理：从一般性的前提出发，通过推导，得出具体陈述或个别结论的过程。逻辑推理的种类按推理过程的思维方向划分，一类是从特殊到一般的推理，推理形式主要有归纳、类比，另一类是从一般到特殊的推理,推理形式主要有演绎。

---
### Exercise 2

:question: **Question 2:** 

阅读图灵关于 AI Turing:1950 的原始论文。在论文中，他讨论了对他提出的企业和他的智能测试的几点反驳意见。哪些反对意见仍然有分量？他的反驳是否有效？你能想到自从他写这篇论文以来，事态发展引发的新的反对意见吗？在论文中，他预测到2000年，计算机将有30%的几率通过五分钟的图灵测试，而不需要熟练的询问器。你认为今天电脑有什么机会？再过50年？
> Read Turing’s original paper on AI Turing:1950 .In the paper, he discusses several objections to his proposed enterprise and his test for intelligence. Which objections still carry weight? Are his refutations valid? Can you think of new objections arising from developments since he wrote the paper? In the paper, he predicts that, by the year 2000, a computer will have a 30% chance of passing a five-minute Turing Test with an unskilled interrogator. What chance do you think a computer would have today? In another 50 years?

:exclamation: **Answer 2:**

图灵的九条反驳意见至今仍有分量且有效，他们分别是：神学论点，“鸵鸟”式论点，数学论点，意识论点，种种能力限制的论点，创新论点，神经系统连续性论点，行为变通性论点，超感知论点。今天的电脑拥有更加强大的算力，比以前更加优秀的深度学习，强化学习等方法。因此如今的电脑在很多方面都能做的比人类还好，比如图像识别的正确率方面。如果再过50年，很难想象，我认为机器的发展速度和学习能力比人类强太多太多了，也许会有颠覆认知东西产生。

---
### Exercise 3

:question: **Question 3:** 

每年的罗布纳奖(Loebner Prize)都会颁发给最接近通过图灵测试的程序。调研最新的罗布纳奖得主。它使用什么技术？它如何推动人工智能的发展？ 注：勒布纳奖已在2020年停止颁发。
> Every year the Loebner Prize is awarded to the program that comes closest to passing a version of the Turing Test. Research and report on the latest winner of the Loebner prize. What techniques does it use? How does it advance the state of the art in AI?

:exclamation: **Answer 3:** 

Stephen Worswick 是目前为止最新的罗布纳奖得主，也获得最多该奖的人。他所开发的聊天机器人 Mitsuku，曾在 2013 年、2016 年、2017 年、2018 年和 2019 年获奖。Mitsuku 是一个非常像人的聊天机器人，它模仿英格兰北部一位 18 岁女性的个性。任何有网络连接的人都可以自由地与 Mitsuku 聊天，问她任何问题，它都会回答。它在模仿人类反应方面做得很好，能够识别现代口语、计算机缩写（如 LOL）和时事。询问她最喜欢的足球队，她会“愉快地”告诉您有关利兹联队的信息。有趣的是，“终结者”是她最喜欢的电影。虽然 Mitsuku 逐年改进，但聊天机器人和文本另一端的人类之间仍然存在明显的差异。Mitsuku 很难识别常见的拼写错误或字母互换。即使它努力与互动的人继续对话，对话也存在一个明显的截止点。

它使用 AIML (Artificial Intelligence Markup Language) 来识别用户输入的关键词和语法，并回复相应的回答。AIML 是一种基于模式匹配的语言，用于定义人工智能与人类之间的对话。它允许开发人员使用类似于编程语言的语法来定义人机对话的模式。 例如，如果有人问“你能吃下一个房子吗？”，它会查询“房子”的性质。发现“made_from”的值是“brick”，就会回答“不能”。

Mitsuku 作为一个高度可定制的聊天机器人，它能够与人类进行多种类型的对话，并且不断改进。这也推动了人工智能在自然语言处理和对话系统方面的发展。 提高自然语言理解能力： Mitsuku 使用 AIML 技术来识别和理解人类语言，这有助于提高 AI 的自然语言理解能力。 提高对话系统的可用性： Mitsuku 可以进行各种类型的对话，并且可以根据用户的需求进行自定义，这有助于提高 AI 对话系统的可用性。 提高对话系统的交互性： Mitsuku 能够与人类进行自然的对话，这有助于提高 AI 对话系统的交互性。

---
### Exercise 4

:question: **Question 4:** 

反射动作（例如从热炉中退缩）是否合理？他们智能吗？
> Are reflex actions (such as flinching from a hot stove) rational? Are they intelligent?

:exclamation: **Answer 4:** 

反射动作是合理的。反射动作通常比经过深思熟虑后采取的较慢的动作更为成功，它属于理性行为的一种方式，而理性行为是理性智能体方法，它也是智能的。

---
### Exercise 5

:question: **Question 5:**  

有一些众所周知的问题是计算机难以解决的，还有一些问题是无法确定的。这是否意味着人工智能是不行的？
> There are well-known classes of problems that are intractably difficult for computers, and other classes that are provably undecidable. Does this mean that AI is impossible?

:exclamation: **Answer 5:**  

虽然有一些众所周知的问题是计算机难以解决的，但从人工智能的诞生（1943-1956），起步发展期（1956-1969），反思发展期（1966-1973），应用发展期-专家系统（1969-1986），神经网络的回归（1986-现在），概率推理和机器学习（1987-现在），大数据（2001-现在），深度学习（2011-现在）。经历这些阶段之后产生的新兴方向，比如自动驾驶、腿足式机器人，自动规划和调度，机器翻译，语音识别，推荐系统，博弈，图像理解，医学，气候科学等等这些，表明人工智能在特定方向进步的速度比人类还快。按照这种发展与进步的速度，人工智能迟早可以解决那些众所周知的问题的。所以这并不意味着人工智能不行。

---
### Exercise 6

:question: **Question 6:**  

假设我们扩展了Evans的SYSTEM程序，使其在标准智商测试中可以获得200分。那么我们会有一个比人类更聪明的程序吗？
> Suppose we extend Evans’s SYSTEM program so that it can score 200 on a standard IQ test. Would we then have a program more intelligent than a human? Explain.

:exclamation: **Answer 6:** 

目前还不能做到一个比人类更聪明的程序，即使其在标准智商测试能获得200分。目前人工智能也只能在单一方面或者某些方面有较出色的表现，在综合方面的学习与表现是远不如人类的。智商测试虽然能达到200分，那也只是在智商测试这一方面很卓越，在其他方面并不能表现都很出色。

--- 
### Exercise 7

:question: **Question 7:** 

sea slug Aplysis 的神经结构得到了广泛的研究(首先是由诺贝尔奖获得者埃里克·坎德尔(Eric Kandel)进行的)，因为它只有大约2万个神经元，其中大多数都很大，很容易操纵。假设Aplysis神经元的周期时间与人类神经元大致相同，那么就每秒内存更新而言，与图1.3中描述的高端计算机相比，其计算能力如何?

> The neural structure of the sea slug Aplysis has been widely studied (first by Nobel Laureate Eric Kandel) because it has only about 20,000 neurons, most of them large and easily manipulated. Assuming that the cycle time for an Aplysis neuron is roughly the same as for a human neuron, how does the computational power, in terms of memory updates per second, compare with the high-end computer described in (Figure 1.3)?

:exclamation: **Answer 7:** 

Aplysis神经结构的计算能力相对于高端计算机来说要低得多。Aplysis只有约2万个神经元，而高端计算机可以拥有数十亿个处理器。因此，Aplysis的计算能力要远远低于高端计算机。

---
### Exercise 8

:question: **Question 8:** 

自省——对一个人内心想法的报告——怎么会不准确呢?我的想法会不会是错的?请讨论。
> How could introspection—reporting on one’s inner thoughts—be inaccurate? Could I be wrong about what I’m thinking? Discuss.

:exclamation: **Answer 8:** 

所谓自省，指一个人内心想法进行思考的过程，是不准确的。原因是自我反省是具有偏见的，这些偏见来自人类生活的方方面面，有对事物客观认知错误的认知偏差，有为了满足大脑趋利避害需求的自我欺骗偏差，有受限于大脑知识容量而忘记部分信息的记忆偏差，也有社会地位偏差，比如社会不同阶级的人无法以理性的角度看同一件事情。综上，我认为人的自我反省是不准确的，但可以在反省的过程中借助工具和他人的力量做到尽量准确。

---
### Exercise 9

:question: **Question 9:** 

以下计算机系统实例是否是人工智能的例子:
* 超市条码扫描器。
* 网络搜索引擎。
* 语音激活的电话菜单。
* 对网络状态作出动态反应的互联网路由算法。

> To what extent are the following computer systems instances of artificial intelligence: - Supermarket bar code scanners. - Web search engines. - Voice-activated telephone menus. - Internet routing algorithms that respond dynamically to the state of the network.

:exclamation: **Answer 9:**

- 超市条码扫描器：不被认为是人工智能的应用，他们只是用来执行特定的程序，将商品上的二维码匹配到固定的信息上，不被认为具有泛化到其他任务的潜力 
- 网络搜索引擎：搜索系统被认为是人工智能的实例，因为它们为根据使用者的查询返回相关结果。它们的设计中包含根据使用者的反馈来学习和优化内部的算法参数，从而改进搜索结果。 
- 语音激活的电话菜单：语音激活系统被认为是人工智能的实例，因为它们具有识别和回应口头命令的能力，可以理解自然语言。 
- 对网络状态作出动态反应的互联网路由算法： 现代的互联网路由算法被认为是人工智能的实例，因为它们能够根据实时网络信息做出决策，并根据反馈不断变化来更好的适应当前的网络条件。

---
### Exercise 10

:question: **Question 10:** 

以下计算机系统在多大程度上是人工智能的实例:
* 超市的条形码扫描器。
* 语音激活的电话菜单。
* Microsoft Word中的拼写和语法纠正功能。
* 对网络状态作出动态反应的互联网路由算法

> To what extent are the following computer systems instances of artificial intelligence: - Supermarket bar code scanners. - Voice-activated telephone menus. - Spelling and grammar correction features in Microsoft Word. - Internet routing algorithms that respond dynamically to the state of the network.

:exclamation: **Answer 10:**

- 超市的条形码扫描器：同1.9 
- 语音激活的电话菜单：同1.9 
- Microsoft Word中的拼写和语法纠正功能：被认为是人工智能算法的实例，可以理解用户输入语句的意思，并找出错误，可以随着反馈更新迭代自己的参数，越变越好，具有学习能力。 
- 对网络状态作出动态反应的互联网路由算法：同1.9

---
### Exercise 11

:question: **Question 11:** 

许多已经提出的认知活动的计算模型涉及相当复杂的数学运算，比如用高斯对图像进行卷积或寻找熵函数的最小值。大多数人（当然还有所有的动物）根本就没有学过这种数学，几乎没有人在大学之前学过，也几乎没有人能够在脑子里计算出一个函数与高斯的卷积。说 "视觉系统"在做这种数学运算，而实际的人却不知道怎么做，这有什么意义呢？

> Many of the computational models of cognitive activities that have been proposed involve quite complex mathematical operations, such as convolving an image with a Gaussian or finding a minimum of the entropy function. Most humans (and certainly all animals) never learn this kind of mathematics at all, almost no one learns it before college, and almost no one can compute the convolution of a function with a Gaussian in their head. What sense does it make to say that the “vision system” is doing this kind of mathematics, whereas the actual person has no idea how to do it?

:exclamation: **Answer 11:**

首先，针对人类理解一张图像或者人类理解眼睛所看到的成像而言，人类经过了漫长且复杂的学习过程，婴儿不可能理解卡车和轿车的区别，在经过了不断的见识增长和经验累积后，成年人区分对卡车和轿车易如反掌，而成年人更多的是依赖经验，即过去学习到的知识。

其次，题目中最后一句话，两者之间有一定的联系。实际的人在大脑中会储存过去学习的知识，相当于已经学习到了网络权重，当人类在看到图像时直接回忆知识，无需在看到图像当场进行知识学习，人脑可以直接进行推理；类比为视觉系统的推理过程，此时视觉系统中的网络权重已经确定，无需再让视觉系统反复进行训练学习。我们也遇到过，不认识的植物，此时我们会先在大脑中搜索、回忆已知的植物类别，匹配我们大脑中已知品种，如果没有已知的品种，我们可能会通过辨识这个植物的叶形、叶尖、脉序等特征，推测它属于哪一个类，哪一个属。

回到人类没有对植物分门归类的时刻，此时世界上的所有植物都没有名字，第一个人是根据什么来区分植物的呢？他可能会先观察植物的叶子，然后根据叶子的形状、大小、颜色等特征，将植物分为不同的类别，然后给这些类别起名字，这样就可以区分不同的植物了。这个过程就是人类学习的过程。然后第一个命名了某个植物的人，将这个植物的名字记录下来，并教授给其他人，相当于其他人直接记录了植物族谱网络的权重。所以经过了一定知识学习和经验累积的人类不需要在脑中进行大量的复杂计算，依靠经验与感性认识就可以处理大部分问题，识得大部分物体。

<div style="color:#409EFF">
最后，对于给定的任务要创造智能体或智能算法时，必须考虑可计算性和易处理性。人类发现红绿蓝可以组成已知世界中的任何一种颜色，不同大小的图像又可以抽象地划分为不同数量的颜色块的组合。采取这种将真实世界抽象为像素块的方法，视觉系统可以模拟人类在大脑中处理图像的感性过程，这一模拟方法的意义体现了人工智能必备的可计算性和易处理性。</div>

---
### Exercise 12

:question: **Question 12:**

一些作者声称，感知和运动技能是智力最重要的部分，而"更高层次"的能力必然是寄生的--这些基本设施的简单附加物。当然，大部分的进化和大部分的大脑都致力于感知和运动技能，而人工智能发现诸如游戏和逻辑推理等任务在很多方面都比在现实世界中的感知和行动要容易。你认为人工智能对高层次认知能力的传统关注是错误的吗？

> Some authors have claimed that perception and motor skills are the most important part of intelligence, and that “higher level” capacities are necessarily parasitic—simple add-ons to these underlying facilities. Certainly, most of evolution and a large part of the brain have been devoted to perception and motor skills, whereas AI has found tasks such as game playing and logical inference to be easier, in many ways, than perceiving and acting in the real world. Do you think that AI’s traditional focus on higher-level cognitive abilities is misplaced?

:exclamation: **Answer 12:**

我认为，感知和运动技能并不是智力最重要的部分，逻辑推理是更加重要的。人工智能的发展受到了人类认知能力的限制，人类认知能力的发展受到了生物进化的限制。在默认了已知世界的全部知识是可以通过逻辑推理得来时，逻辑推理对于生物进化，人类发展乃至人工智能的发展是至关重要的。

---
### Exercise 13

:question: **Question 13:**

为什么进化会倾向于形成做事理性的系统？这样的系统是为了实现什么目标而设计的呢?
> Why would evolution tend to result in systems that act rationally? What goals are such systems designed to achieve?

:exclamation: **Answer 13:**

首先，进化倾向于产生理性行为的系统，因为理性行为是基于目前信息收益最大化的行为，往往可以增加收益。其次，不论是自然世界中系统的进化和计算机程序系统的进化，都有特定的内在规律和被设置好的目标，只不过自然世界的规律是达尔文进化论，目标是生存和繁衍，计算机的规律是各种科学原理，目标是人类所设定的各种任务，如识别，分类和各种自然语言处理子任务。所以，我们观察到进化成功的系统，都满足了这些理性的任务，所以我们会认为进化最终会趋于理性，这也是幸存者偏差的一种。

---
### Exercise 14

:question: **Question 14:**

人工智能是科学，还是工程?或者两者都不是?解释一下。
> Is AI a science, or is it engineering? Or neither or both? Explain.

:exclamation: **Answer 14:**

人工智能即使科学又是工程。人工智能是科学这点毋庸置疑，任何智能系统背后都有无数的数学原理和科学理论。人工智能同时也是工程，因为它涉及将这些科学知识转为实际中的应用，以创建和设计可以使用人工智能的系统，人工智能算法的训练和部署等等。

---
### Exercise 15

:question: **Question 15:**

“当然，计算机不可能是智能的，他们只能按照程序员的指示去做。”后一种说法是正确的吗？这是否意味着前者也是正确的？
> “Surely computers cannot be intelligent—they can do only what their programmers tell them.” Is the latter statement true, and does it imply the former?

:exclamation: **Answer 15:**

后一种说法并不正确，按照程序员的指示去做只是其中的一种方式，在计算机进行无监督学习的过程，程序员是没有明确标注数据，也就是相当于没有明确的指示。但是计算机也能做出智能分类，这也就验证了前者所说计算机不可能是智能的是错误的。

---
### Exercise 16

:question: **Question 16:**

“动物当然不可能是聪明的，它们只能按照基因的指示行事。”后一种说法是正确的吗？这是否意味着前者也是正确的？
> “Surely animals cannot be intelligent—they can do only what their genes tell them.” Is the latter statement true, and does it imply the former?

:exclamation: **Answer 16:**

后一种说法不正确，基因能决定一部分，但是个体在环境当中也能产生适应环境指示的行为。比如狗看到肉骨头会去吃，但是经过环境当中人的训练，人可以告诉狗不要吃肉骨头，而这时候狗也就不完全受基因影响而一定马上会去吃肉骨头。当然，这也不能说明动物是不聪明的，反而大部分动物都很聪明。比如狼有很好的团队合作，这是很聪明的。这些既有先天基因决定的一部分，也有后天学习的一部分。

---
### Exercise 17

:question: **Question 17:**

“当然，动物、人类和计算机不可能是智能的，它们只能做物理定律告诉它们的组成原子做的事情。”后一种说法是正确的吗？这是否意味着前者也是正确的？
> “Surely animals, humans, and computers cannot be intelligent—they can do only what their constituent atoms are told to do by the laws of physics.” Is the latter statement true, and does it imply the former?

:exclamation: **Answer 17:**

后一种说法不正确，人类不仅能做已发现的物理定律范围内的事情，人类还能探索更多尚未发现科学。计算机不仅能做现在已有程序所能执行的范围内的事情，还能进行深度学习，做更多自己以前不能做且未知的事情。动物的进化，也证明了动物也是可以做到之前所受定律限制而不能做的事情。这意味着动物、人类、计算机都可能是智能的。

---
### Exercise 18

:question: **Question 18:**

查阅人工智能文献，查阅以下任务目前是否可以由计算机解决：

* 打一局像样的乒乓球（Ping-Pong）。
* 在埃及开罗的市中心开车。- 在加利福尼亚的维克多维尔开车。
* 在市场上买一个星期的杂货。
* 在网上购买一周的食品杂货。
* 在有竞争力的水平上打一局体面的桥牌。
* 发现并证明新的数学定理。
* 写一个有意的搞笑故事。
* 在一个专门的法律领域提供合格的法律建议。
* 将英语口语实时翻译成瑞典语口语。
* 进行一次复杂的外科手术。

>  Examine the AI literature to discover whether the following tasks can currently be solved by computers: - Playing a decent game of table tennis (Ping-Pong). - Driving in the center of Cairo, Egypt. - Driving in Victorville, California. - Buying a week’s worth of groceries at the market. - Buying a week’s worth of groceries on the Web. - Playing a decent game of bridge at a competitive level. - Discovering and proving new mathematical theorems. - Writing an intentionally funny story. - Giving competent legal advice in a specialized area of law. - Translating spoken English into spoken Swedish in real time. - Performing a complex surgical operation.

:exclamation: **Answer 18:**

* 打一局像样的乒乓球（Ping-Pong）。可以打乒乓球的人工智能机器人已经被开发出来，有些甚至能够击败人类选手。

* 在埃及开罗的市中心开车：AI驱动的自动驾驶汽车目前无法在开罗等许多大城市复杂混乱的交通状况下行驶。

* 在加利福尼亚的维克多维尔开车：AI驱动的自动驾驶汽车已经在加州道路。然而，根据自主程度的不同可以处理的道路情况也不同。

* 在市场上买一个星期的杂货：AI驱动的购物助理已经开发出来，但它们目前还是为推荐，搜索和广告等业务而服务，虽然还没有发展到可以为个人定制的程度，但可以通过自动下单的方式在市场上实现杂货购买。

* 在网上购买一周的食品杂货。人工智能驱动的购物助手已经被开发出来，但它们目前还是为推荐，搜索和广告等业务而服务，虽然还没有发展到可以为个人定制的程度，但可以通过自动下单的方式在网上实现食品购买。

* 在有竞争力的水平上打一局体面的桥牌：人工智能桥牌系统已经被开发出来，可以打出很高的水平，有些甚至能够击败人类玩家。

* 发现并证明新的数学定理。人工智能驱动的定理证明器还不能发现和证明新的数学定理。

* 写一个有意的搞笑故事：人工智能驱动的自然语言生成系统可以生成文本，可以根据训练数据中的内容生成一些故意搞笑的故事。

* 在一个专门的法律领域提供合格的法律建议：人工智能支持的法律咨询系统可以协助进行法律研究和文件分析，但它们还不具有足够的专业性。

* 将英语口语实时翻译成瑞典语口语：人工智能语音翻译系统可以实时翻译口语，翻译的准确性和流畅性可能会因具体系统和训练时使用的数据量而异。

* 进行一次复杂的外科手术：人工智能驱动的手术机器人还不能像人类外科医生那样执行复杂的外科手术。

---
### Exercise 19

:question:  **Question 19:**

对于目前无法解决的任务，尝试找出困难所在，并预测何时能克服这些困难。
> For the currently infeasible tasks, try to find out what the difficulties are and predict when, if ever, they will be overcome.

:exclamation: **Answer 19:**

目前无法解决的任务如下。
* 在埃及开罗的市中心开车：这需要汽车能够在复杂的和变化的交通环境中运行，并能够应对各种各样的道路状况和行人。目前有相关研究通过预测算来来模拟路况的变化和行人举动，有望在近期实现。
* 发现并证明新的数学定理。人工智能驱动的定理证明器还不能发现和证明新的数学定理。需要更深刻的符号理解和语言处理中的逻辑思考水平，有望在下一次技术井喷之后实现。
* 进行一次复杂的外科手术：人工智能驱动的手术机器人还不能像人类外科医生那样执行复杂的外科手术，需要很多相关学科的支持，如柔性纳米材料机器人的实现。

---
### Exercise 20

:question:  **Question 20:**

人工智能的各个子领域都通过定义一个标准任务并邀请研究人员做到最好来举办比赛。这方面的例子包括DARPA的机器人汽车大挑战，国际规划竞赛，Robocup机器人足球联赛，TREC信息检索活动，以及机器翻译和语音识别的竞赛。调查这些竞赛中的五项，并描述多年来取得的进展。这些比赛在多大程度上推动了人工智能技术的发展？它们在多大程度上因为把精力从新的想法中抽走而损害了这个领域？
> Various subfields of AI have held contests by defining a standard task and inviting researchers to do their best. Examples include the DARPA Grand Challenge for robotic cars, the International Planning Competition, the Robocup robotic soccer league, the TREC information retrieval event, and contests in machine translation and speech recognition. Investigate five of these contests and describe the progress made over the years. To what degree have the contests advanced the state of the art in AI? To what degree do they hurt the field by drawing energy away from new ideas?

:exclamation: **Answer 20:**

1. DARPA机器人汽车大挑战: 该比赛旨是为了推动汽车自动驾驶技术的发展，在无人驾驶汽车上完成复杂的道路挑战。每届比赛中都提高了自动驾驶汽车在道路上的安全性和可靠性，对该领域是有正向帮助的。
2. 国际规划竞赛: 该比赛是为了推动自主规划技术的发展，开发能够解决各种不同类型问题的规划系统。每届比赛中都提高了规划系统在复杂问题上的效率和准确性，对该领域是有正向帮助的。
3. Robocup机器人足球联赛: 该比赛是为了推动机器人智能和协作技术的发展，开发能够在人类与机器人之间进行足球比赛的机器人。每届比赛都提高了机器人在比赛中的运动能力和协作能力，对该领域是有正向帮助的。。
4. TREC信息检索活动: 该比赛是为了推动信息检索技术的发展，开发能够在大型文本数据库中快速准确地检索信息的系统。每届比赛都会大幅提高信息检索系统的准确性和速度，对信息检索领域是有正向帮助的。
5. 机器翻译和语音识别的竞赛: 这些比赛是为了推动机器翻译和语音识别技术的发展，开发能够翻译或识别语音的系统。每届比赛中都会大幅提高了机器翻译和语音识别系统的准确性和速度。

