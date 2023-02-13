# Meta Label Correction 论文阅读

## 1.文章背景介绍

训练神经网络的时候，噪声标签是个很麻烦的问题，因为非常容易导致过拟合。

本文同样也是讨论噪声标签的问题，一开始作者就阐述了近些年有些样本重新赋权的方法取得了很好的效果。

解决这个问题通常有两种做法：一个是使用co-teaching或curriculum learning的方式，从噪声数据中挑选出正确的样本；另一个就是使用重新赋权（re-weight），这种方式可以保留全部的数据。这其中有人使用了元学习的方法，

但作者认为，这种重新赋权（re-weight）的方式有局限性，它只能降低或提升权重来控制样本对学习的贡献。

另一种可以替代的方案是基于一些假设的前提来纠正噪声标签。

本文解决问题的方式并不是重新赋权，而是从另外一个角度入手：既然标签本身有错误，那么是不是可以用自动的方式把错误标签纠正过来呢？

标签的噪声有多种产生方式，比如一些并不专业的人进行标记时难免出错，还有些是基于启发式和用户交互信号（heuristics or user interaction signals）的自动标注。

纠正标签的目的，一个是可以提高干净标签的权重，另一个是把错误的标签改过来。

以往纠正错误标签的假设都基于这个假设：1.标签损坏矩阵（estimating a label corruption matrix）2.我们的模型是基于符合这个矩阵规律的数据来训练的。这个假设成立的条件比较苛刻，通常需要假设噪声标签只依赖于真实标签，与数据本身无关。

本文提出了一个从噪声数据中纠正标签的元学习方法，取名为MLC（meta label correction，元标签纠正）。把纠正标签的过程看作一个元过程（meta-process）。

利用被元学习模型所纠正后的标签来训练成一个新的预测模型。

元学习模型和常规分类模型同时进行训练，在一个双层的优化过程（bi-level optimization procedure）里。这个可以最大化模型在干净数据集上的表现，通过更正错误标签。MLC同时发挥了重新赋权和纠正标签的双重优点。与以往的纠正标签的方法相比，它不需要基于对噪声数据本身的假设，而是直接训练一个纠正模型。







### 名词解释

| 名词                | 中文（自己翻译的） | 解释                                                         | 图解 |
| ------------------- | ------------------ | ------------------------------------------------------------ | ---- |
| Uniform label noise | 均匀噪声           | 一个数据集有C个类别，真实的标签y有$\rho/C$的概率会被错误地变成另一个可能的类别$y'$，有$1-\rho$的概率保持不变。 |      |
| Flipped label noise | 翻转噪声           | 一个数据集有C个类别，真实的标签y有$\rho$的概率会被错误地变成任何其他的类别（共$C-1$个），有$1-\rho$的概率保持不变。 |      |
|                     |                    |                                                              |      |



## 2.相关工作



## 3.算法设计

### 3.1 算法描述

我们假设有两组数据：一个干净的、可信赖的小数据集，一个含有噪声标签的大数据集。为什么这样假设呢，因为花钱请专家标注数据比较贵，所以干净的数据相比于噪声的数据要小很多。这个时候，如果直接在小数据集上训练并不是最好的选择，这样很容易过拟合。而直接在大的数据集上训练也不好，模型会直接把噪声数据全部学到了。

这时候，我们通过训练一个元模型和一个主模型，前者纠正噪音标签，后者把修正后的标签拿来训练，让这2个模型互相加强。

干净数据集，我们称作$D=\{x,y\}^m$;噪声数据集，我们称作$D'=\{x,y\}^M$,m远小于M。

我们建立了一个标签纠正网络（label correction network），接受一组噪声数据及其标签作为输入，一个新的纠正后的标签作为输出。

LCN的目标是形成一个参数为$\alpha$的函数：$y_c=g_{\alpha}\left( h(x),y'\right)$，其中$y_c$是纠正后的标签，它是一个软标签。而$h(x),y'$是一组噪声标签的数据。

主模型$f$目标是生成一个参数为$w$的函数：$y=f_w(x)$。

显然，这两个方法只能各自为战。我们通过一个双层优化的方式把他们连接在一起，这个是解一个优化问题：
$$
\min_\alpha\mathrm{E}_{(x,y)\in D}l\left(y,f_{w^{*}_\alpha(x)}\right)
$$
$s.t.$  
$$
w^*_\alpha= \argmin
$$
 $$

### 3.2 代码实现

读取数据：

#### 取MW-Net：

主函数：

```python
prepare_data_mwnet(gold_fraction, #分割比例
                   corruption_prob, #破坏标签的概率
                   corruption_type, #破坏标签的类别
                   args)
```

返回值：

```python
return train_gold_loader, #载入的是DataLoader，读取元学习数据集
	   train_silver_loader, #载入的是DataLoader，读取的是常规训练集
       valid_loader, #载入的是DataLoader，读取的是常规验证集
       test_loader, #载入的是DataLoader，读取的是常规测试集
       num_classes #返回数据集包含的类别数量
```

主要的实现过程：



#### 取MLC（本文实现的算法）：

主函数：

```python
prepare_data_mwnet(gold_fraction, #分割比例
                   corruption_prob, #破坏标签的概率
                   corruption_type, #破坏标签的类别
                   args)
```

返回值：

```python
return train_gold_loader, #载入的是DataLoader，读取元学习数据集
	   train_silver_loader, #载入的是DataLoader，读取的是常规训练集
       valid_loader, #载入的是DataLoader，读取的是常规验证集
       test_loader, #载入的是DataLoader，读取的是常规测试集
       num_classes #返回数据集包含的类别数量
```

#### 数据集处理：

制造均匀噪声：

```python
return mixing_ratio * #这个应该是几率
	   np.full((num_classes, num_classes),1 / num_classes)
	   + (1 - mixing_ratio) * np.eye(num_classes)
```

制造翻转噪声：

```python
def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C
```

#### 建立模型：

```python
if dataset in ['cifar10', 'cifar100']:
	from CIFAR.resnet import resnet32
	# main net
    model = resnet32(num_classes)
    main_net = model

    # meta net
    hx_dim = 64 #0 if isinstance(model, WideResNet) else 64 # 64 for resnet-32
    meta_net = MetaNet(hx_dim, cls_dim, 128, num_classes, args)
```

主要的骨架模型：



```python
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, return_h=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)
        if return_h:
            return out, hidden
        else:
            return out
```



元模型：

```python
class MetaNet(nn.Module):
    def __init__(self, hx_dim, cls_dim, h_dim, num_classes, args):
        super().__init__()
    	self.args = args

        self.num_classes = num_classes        
        self.in_class = self.num_classes 
        self.hdim = h_dim
        self.cls_emb = nn.Embedding(self.in_class, cls_dim) #这里开始构建了嵌入

        in_dim = hx_dim + cls_dim #

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, num_classes + int(self.args.skip), bias=(not self.args.tie)) 
        )

        if self.args.sparsemax:
            from sparsemax import Sparsemax
            self.sparsemax = Sparsemax(-1)

        self.init_weights()

        if self.args.tie:
            print ('Tying cls emb to output cls weight')
            self.net[-1].weight = self.cls_emb.weight

    def init_weights(self):
        nn.init.xavier_uniform_(self.cls_emb.weight)
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[2].weight)
        nn.init.xavier_normal_(self.net[4].weight)

        self.net[0].bias.data.zero_()
        self.net[2].bias.data.zero_()

        if not self.args.tie:
            assert self.in_class == self.num_classes, 'In and out classes conflict!'
            self.net[4].bias.data.zero_()

    def get_alpha(self):
        return self.alpha if self.args.skip else torch.zeros(1)

    def forward(self, hx, y):
        bs = hx.size(0)

        y_emb = self.cls_emb(y)
        hin = torch.cat([hx, y_emb], dim=-1)

        logit = self.net(hin)

        if self.args.skip:
            alpha = torch.sigmoid(logit[:, self.num_classes:])
            self.alpha = alpha.mean()
            logit = logit[:, :self.num_classes]

        if self.args.sparsemax:
            out = self.sparsemax(logit) # test sparsemax
        else:
            out = F.softmax(logit, -1)

        if self.args.skip:
            out = (1.-alpha) * out + alpha * F.one_hot(y, self.num_classes).type_as(out)

        return out
```



模型运行：

```python
def run():
    corruption_fnctn = uniform_mix_C if args.corruption_type == 'unif' else flip_labels_C
    filename = '_'.join([args.dataset, args.method, args.corruption_type, args.runid, str(args.epochs), str(args.seed), str(args.data_seed)])

    results = {}

    gold_fractions = [0.02] 

    if args.gold_fraction != -1:
        assert args.gold_fraction >=0 and args.gold_fraction <=1, 'Wrong gold fraction!'
        gold_fractions = [args.gold_fraction]

    corruption_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if args.corruption_level != -1: # specied one corruption_level
        assert args.corruption_level >= 0 and args.corruption_level <=1, 'Wrong noise level!'
        corruption_levels = [args.corruption_level]

    for gold_fraction in gold_fractions:
        results[gold_fraction] = {}
        for corruption_level in corruption_levels:
            # //////////////////////// load data //////////////////////////////
            # use data_seed her
            gold_loader, silver_loader, valid_loader, test_loader, num_classes = get_data(args.dataset, gold_fraction, corruption_level, corruption_fnctn)
            
            # //////////////////////// build main_net and meta_net/////////////
            main_net, meta_net = build_models(args.dataset, num_classes)
            
            # //////////////////////// train and eval model ///////////////////
            exp_id = '_'.join([filename, str(gold_fraction), str(corruption_level)])
            test_acc, baseline_acc = train_and_test(main_net, meta_net, gold_loader, silver_loader, valid_loader, test_loader, exp_id)
        
            results[gold_fraction][corruption_level] = {}
            results[gold_fraction][corruption_level]['method'] = test_acc
            results[gold_fraction][corruption_level]['baseline'] = baseline_acc
            logger.info(' '.join(['Gold fraction:', str(gold_fraction), '| Corruption level:', str(corruption_level),
                  '| Method acc:', str(results[gold_fraction][corruption_level]['method']),
                                  '| Baseline acc:', str(results[gold_fraction][corruption_level]['baseline'])]))
            logger.info('')


    with open('out/' + filename, 'wb') as file:
        pickle.dump(results, file)
    logger.info("Dumped results_ours in file: " + filename)
```



## 4.实验设计

设定了一个概率$\rho$来破坏现有的数据集，训练过程中，模型并不知道这个概率的存在，对数据集的破坏也是随机生成的。

挑选了两个最具有代表性的SOTA对手拿来对比：基于纠正标签的GLC，基于重新赋权的Meta-Weight-Net。



### 4.1 图片分类任务

| 数据集      | 元学习数据集                     | 使用的模型              | 制造翻转噪声 |
| ----------- | -------------------------------- | ----------------------- | ------------ |
| CIFAR-10    | 从中取1000张图片                 | ResNet-32               |              |
| CIFAR-100   | 从中取1000张图片                 | ResNet-32               |              |
| Clothing 1M | 使用这个数据集自带的干净数据子集 | ResNet-50（pretrained） |              |





### 4.2 文本分类任务

## 5.个人点评
