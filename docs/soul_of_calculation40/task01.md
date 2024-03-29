# Task01 算法的规范化和量化度量、大数和数量级的概念

## 1 算法的规范化和量化度量

- 计算机的研制：ENIAC主要用于解决长程火炮过程中的计算问题，需要通过修改线路，完成其他方向的计算；EDVAC提供了新的设计方案，是世界上第一台程序控制的通用电子计算机
- 冯·诺依曼体系结构：将计算机分为了软硬件两部分，由于早期软件算法简单，大多数用于科学计算，针对商业需要计算机算法来弥补
- 高德纳：
    - 计算机算法分析鼻祖，提出评估计算机算法的标准
    - 编写《计算机程序设计艺术》
    - 最年轻的图灵奖获得者
    - 编写Tex软件
    - 硅谷地区众多图灵奖获得者中名气最大、最会编程的人

## 2 大数和数量级的概念

- 大数的认知
    - 财务软件的故事：最初对一次账几秒，当数据量大10倍之后，对账时间超过了十倍
    - 两个原始部落的酋长的故事：他们所拥有的东西很少超过三个，比三多的范围就用“许多”来形容
    - 一亿元的认知：对大部分人来讲，一亿元人民币等于财富自由，等于无穷大
- 算法复杂度严格量化衡量（高德纳的思想）
  1. 数据量特别大近乎无穷大时的算法效率
  2. 考虑不随数据量变化和随数据量变化的因素：考虑$N$趋近于无穷大时和$N$相关的那部分来讨论算法复杂度
  3. 算法复杂度中省略数量级低的那一部分

## 3 课后思考题

### 3.1 思考题1.1

&emsp;&emsp;世界上还有什么产品类似于计算机，是软硬件分离的？

**解答：**

- 汽车：硬件是发动机、底盘、车身和电气设备；软件是智能AI系统，比如倒车影像位置识别，车身感应器等
- 电视：硬件是外壳、液晶面板、挂架或底座、电路系统；软件是搭载在上面的智能电视软件
- 信件分拣系统：硬件是邮件传输带、分拣手臂、电路控制设备；软件是信件地区智能识别

### 3.2 思考题1.2

&emsp;&emsp;如果一个程序只运行一次，在编写它的时候，你是采用最直观但是效率较低的算法，还是依然寻找复杂度最优的算法？

**解答：**

&emsp;&emsp;如果一个程序只运行一次，在编写它的时候，依然需要寻找复杂度最优的算法。因为该程序可能只运行一次，但是该程序中包含的算法可能会重用到其他场景下。