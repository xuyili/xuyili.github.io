# 对象

> Object对象是Javascript中的基本数据类型，笔者将围绕对象中三种创建方式分别在每种创建方式中通过实例介绍对象的一些特性及需注意的问题。

#### 快速导航

* [对象的三种类型介绍](#对象的三种类型介绍)
* [创建对象的四种方法](#创建对象的四种方法)
    - [对象字面量创建](#对象字面量创建)
    - [使用new关键字构造形式创建](#使用new关键字构造形式创建)
    - [对象的create方法创建](#对象的create方法创建)
    - [原型prototype创建](#原型prototype创建)
* [对象属性描述符](#对象属性描述符)
* [对象的存在性检测](#对象的存在性检测)
* [对象引用传递](#对象引用传递)
    - [引用类型示例比较](#引用类型示例分析)
    - [对象copy三种实现方式](#对象copy)

#### 面试指南

* ``` 什么是引用传递？{} == {} 是否等于true  ```，参考：[对象引用类型示例分析](#引用类型示例分析)
* ``` 如何编写一个对象的深度拷贝函数？ ```，参考：[对象copy实现](#对象copy实现)
* ``` new操作符具体做了哪些操作，重要知识点！ ```，参考：[使用new关键字构造形式创建](#使用new关键字构造形式创建)
    ```
        var p = [];
        var A = new Function();
        A.prototype = p;
        var a = new A;
        a.push(1);
        console.log(a.length);
        console.log(p.length);
    ```

## 对象的三种类型介绍:

* 内置对象，（String、Number、Boolean、Object、Function、Array）
* 宿主对象，由Javascript解释器所嵌入的宿主环境定义的，表示网页结构的HTMLElement对象均是宿主对象，也可以当成内置对象
* 自定义对象

## 创建对象的四种方法:

* 对象字面量 ``` var obj = { a: 1 } ```
* 使用new关键字构造形式创建 ``` var obj = new Object({ a: 1}) ```
* 原型（prototype）创建
* ES5的Object.create() 方法创建

## 对象字面量创建

#### 对象字面量是由若干个键／值对组成的映射表，整个映射表用{}包括起来

```js
var obj = { a: 1 };

console.log(obj.a);
```

#### 在ES6中增加了可计算属性名

这在一些业务场景中，如果key是预先不能定义的，可以向下面传入变量或者值进行动态计算

```js
var variable = 2;
var obj = {
	[1 + variable]: '我是一个可计算属性名'
}

console.log(obj); // {3: "我是一个可计算属性名"}
```

#### 对象的内容访问

对象值的存入方式是多种多样的，存入在对象容器中的是这些属性的名称，学过C的同学可以想象一下指针的引用，在js中可以理解为对象的引用。内容访问可以通过以下两种符号:

* ``` . ``` 指属性访问
* ``` [] ``` 指键访问

注意：对象中属性名永远必将是字符串，obj[2]看似2是整数，在对象属性名中数字是会被转换为字符串的

```js
var obj = {
	'a': '属性访问',
	2: '键访问'
}

console.log(obj.a); // 属性访问
console.log(obj[2]); // 键访问
```

## 使用new关键字构造形式创建

先介绍下new操作符构造对象的整个过程，这个很重要，明白之后有助于对后续的理解

#### new操作符构造对象过程

* 创建一个全新的对象
* 新对象会被执行prototype操作（prototype之后会写文章专门进行介绍，感兴趣的童鞋可以先关注下）
* 新对象会被绑定到函数调用的this
* 如果函数没有返回新对象，new表达式中的函数调用会自动返回这个新对象（对于一个构造函数，即使它内部没有return，也会默认返回return this）

看一道曾经遇到的面试题，如果在看本篇文章介绍之前，你能够正确理解并读出下面语句，那么恭喜你对这块理解很透彻

```js
var p = [2, 3];
var A = new Function();
    A.prototype = p;

console.log(A.prototype)

var a = new A;

console.log(a.__proto__)

a.push(1);

console.log(a.length); // 3
console.log(p.length); // 2
```

``` new A ``` 时发生了什么?

1. 创建一个新的对象obj

``` var obj = {} ```

2. 新对象执行prototype操作，设置新对象的_proto_属性指向构造函数的A.prototype

``` obj._proto_ = A.prototype ```

3. 构造函数的作用域（this）赋值给新对象

``` A.apply(obj) ```

4. 返回该对象

上面示例中实例a已经不是一个对象，而是一个数组对象，感兴趣的童鞋可以在电脑上操作看下 ``` A.prototype ``` 和 ``` a.__proto__ ``` 的实际输出结果

#### new操作符创建数组对象

数组属于内置对象，所以可以当作一个普通的键/值对来使用。

```js
var arr = new Array('a', 'b', 'c'); // 类似于 ['a', 'b', 'c']

console.log(arr[0]); // a
console.log(arr[1]); // b
console.log(arr[2]); // c 
console.log(arr.length); // 3

arr[3] = 'd';
console.log(arr.length); // 4
```

## 对象的create方法创建

Object.create(obj, [options])方法是ECMAScript5中定义的方法

* ``` obj ``` 第一个参数是创建这个对象的原型
* ``` options ``` 第二个为可选参数，用于描述对象的属性

#### null创建一个没有原型的新对象

```js 
var obj = Object.create(null)

console.log(obj.prototype); // undefined
```

#### 创建一个空对象

以下 ``` Object.create(Object.prototype) ``` 等价于 ``` {} ``` 或 ``` new Object() ```

```js
var obj = Object.create(Object.prototype)

console.log(obj.prototype); // {constructor: ƒ, __defineGetter__: ƒ, __defineSetter__: ƒ, hasOwnProperty: ƒ, __lookupGetter__: ƒ, …}
```

#### 创建原型对象

```js
var obj = Object.create({ a: 1, b: 2 })

console.log(obj.b); // 2 
```

## 原型prototype创建

除了 ```null``` 之外的每一个对象都从原型继承属性，关于javascript的原型之后会有一篇文章进行讲解，本次主要讨论对象的一些内容，所以在这里不做过多讨论

* new Object或者{}创建的对象，原型是 Object.prototype
* new Array创建的对象，原型是 Array.prototype
* new Date创建的对象，原型是 Date.prototype

## 对象属性描述符

ES5之后才拥有了描述对象检测对象属性的方法

* 属性描述符含义
    * ``` {value: 1, writable: true, enumerable: true, configurable: true} ```
    * ``` value ``` 属性值
    * ``` writable ``` 属性值是否可以修改
    * ``` enumerable ``` 是否希望某些属性出现在枚举中
    * ``` configurable ``` 属性是否可以配置，如果是可配置，可以结合 ``` Object.defineProperty() ``` 方法使用

* Object.getOwnPropertyDescriptor(obj, prop)
    * 获取指定对象的自身属性描述符
    * ``` obj ``` 属性对象
    * ``` prop ``` 属性名称

```js
var obj = { a: 1 }
var propertyDesc = Object.getOwnPropertyDescriptor(obj, 'a');

console.log(propertyDesc); // {value: 1, writable: true, enumerable: true, configurable: true}
```

* Object.defineProperty(obj, prop, descriptor)
    * 该方法会直接在一个对象上定义一个新属性，或者修改一个已经存在的属性， 并返回这个对象
    * ``` obj ``` 属性对象
    * ``` prop ``` 属性名称

```js
var obj = { a: 1 };

Object.defineProperty(obj, 'a', {
    writable: false, // 不可写
	configurable: false, // 设置为不可配置后将无法使用delete 删除
})

obj.a = 2;

console.log(obj.a); // 1

delete obj.a;

console.log(obj.a); // 1 
```

* Object.preventExtensions(obj)
    * 禁止一个对象添加新的属性
    * ``` obj ``` 属性对象

```js
var obj = { a: 1 };

Object.preventExtensions(obj)

obj.b = 2;

console.log(obj.b); // undefined
```

## 对象的存在性检测

区分对象中的某个属性是否存在

#### 操作符in检查

in操作符除了检查属性是否在对象中存在之外还会检查在原型是否存在

```js
var obj = { a: 1 };

console.log('a' in obj); // true
```

#### hasOwnProperty

```js
var obj = { a: 1 };

console.log(obj.hasOwnProperty('a')); // true
```


## 对象引用传递

> 对象属于引用类型是属性和方法的集合。引用类型可以拥有属性和方法，属性也可以是基本类型和引用类型。

> javascript不允许直接访问内存中的位置，不能直接操作对象的内存空间。实际上操作的是对象的引用，所以引用类型的值是按引用访问的。准确说引用类型的存储需要内存的栈区和堆区(堆内存)共同完成，栈区内保存变量标识符和指向堆内存中该对象的指针(也可以说该对象在堆内存中的地址)。

#### 引用类型示例分析

1. 引用类型比较

引用类型是按照引用访问的，因此对象(引用类型)比较的是堆内存中的地址是否一致，很明显a和b在内存中的地址是不一样的。

```javascript
const a = {};
const b = {};

a == b //false

```

2. 引用类型比较

下面对象d是对象c的引用，这个值d的副本实际上是一个指针，而这个指针指向堆内存中的一个对。因此赋值操作后两个变量指向了同一个对象地址，只要改变同一个对象的值另外一个也会发生改变。

```javascript
const c = {};
const d = c;

c == d //true

c.name = 'zhangsan';
d.age = 24;

console.log(c); //{name: "zhangsan", age: 24}
console.log(d); //{name: "zhangsan", age: 24}
```

#### 对象copy实现

-  利用json实现

可以利用JSON，将对象先序列化为一个JSON字符串，在用JSON.parse()反序列化，可能不是一种很好的方法，但能适用于部分场景

```js
const a = {
    name: 'zhangsan',
    school: {
        university: 'shanghai',
    }
};

const b = JSON.parse(JSON.stringify(a));

b.school.university = 'beijing';

console.log(a.school.university); // shanghai
console.log(b.school.university); // beijing
```
- es6内置方法

ES6内置的 ``` Object.assign(target,source1,source2, ...) ``` ，第一个参数是目标参数，后面是需要合并的源对象可以有多个，后合并的属性（方法）会覆盖之前的同名属性（方法），需要注意 ``` Object.assign() ``` 进行的拷贝是浅拷贝

```js
const obj1 = {a: {b: 1}};
const obj2 = Object.assign({}, obj1);
 
obj2.a.b = 3;
obj2.aa = 'aa';

console.log(obj1.a.b) // 3
console.log(obj2.a.b) // 3

console.log(obj1.aa) // undefined
console.log(obj2.aa) // aa
```

- 实现一个数组对象深度拷贝

> 对于下面这样一个复杂的数组对象，要做到深度拷贝(采用递归的方式)，在每次遍历之前创建一个新的对象或者数组，从而开辟一个新的存储地址，这样就切断了引用对象的指针联系。

```js
/**
 * [copy 深度copy函数]
 * @param { Object } elments [需要赋值的目标对象]]
 */
function copy(elments){
    //根据传入的元素判断是数组还是对象
    let newElments = elments instanceof Array ? [] : {};

    for(let key in elments){
        //注意数组也是对象类型，如果遍历的元素是对象，进行深度拷贝
        newElments[key] = typeof elments[key] === 'object' ? copy(elments[key]) : elments[key];
    }

    return newElments;
}
```

需要赋值的目标对象

```js
const a = {
    name: 'zhangsan',
    school: {
        university: 'shanghai',
    },
    hobby: ['篮球', '足球'],
    classmates: [
        {
            name: 'lisi',
            age: 22,
        },
        {
            name: 'wangwu',
            age: 21,
        }
    ]
};
```

测试验证，复制出来的对象b完全是一个新的对象，修改b的值，不会在对a进行影响。

```js
const b = copy(a);

b.age = 24;
b.school.highSchool = 'jiangsu';
b.hobby.push('🏃');
b.classmates[0].age = 25;

console.log(JSON.stringify(a)); 
//{"name":"zhangsan","school":{"university":"shanghai"},"hobby":["篮球","足球"],"classmates":[{"name":"lisi","age":22},{"name":"wangwu","age":21}]}
console.log(JSON.stringify(b));

//{"name":"zhangsan","school":{"university":"shanghai","highSchool":"jiangsu"},"hobby":["篮球","足球","🏃"],"classmates":[{"name":"lisi","age":25},{"name":"wangwu","age":21}],"age":24}
```


本次重点是对象的一些总结和探讨，关于原型prototype还有很多内容要讲，文章中有的地方有提到，但没有做过多的阐述，之后会写一篇文章专门进行Javascript原型的讨论，欢迎关注！