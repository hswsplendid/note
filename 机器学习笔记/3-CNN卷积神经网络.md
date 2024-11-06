# 李宏毅机器学习

## 课程链接

1. 网课https://www.bilibili.com/video/BV1Wv411h7kN
2. 课件和资料Github汇总版：https://github.com/Fafa-DL/Lhy_Machine_Learning
3. 公众号【啥都会一点的研究生】课件资料：https://pan.baidu.com/s/1agZm-kXjF4aWH_4lRBv-Sg 

提取码：5b1w



[TOC]



## 第3节课-Convolutional Neural Network

### image classification

影响辨识输入图片时一般保证the same size（图片大小）

输出y^ 的维度表示它能识别多少种类别，y‘是真实输出

<img src="图片/3/1.jpg" style="zoom: 50%;" />

一张图片是三维的tensor（一个维是宽，一个是高，一个是channel的数目）

**tensor 张量**（维度大于2的矩阵

**3个channel** （代表RGB 3个颜色）

每个像素点在都有三个通道的数值，把它们拉直成一个**向量**，作为输入

一张图片是100x100x3个数字，每一个数字代表的是**某一个位置的pixel**（像素）的某个颜色的强度，也就是是每个pixel有三个数字（深度，从外往里数，如下图）

<img src="图片/3/2.jpg" style="zoom:50%;" />

参数过多（weight过多），弹性大，但是容易出现overfitting

比如这张图有100x100x3x1000个weight

**不用做到fully connected**，也就是每一个neuron和input的每一个dimension都有一个weight

<img src="图片/3/3.jpg" style="zoom:50%;" />

### 简化处理的solution--CNN

##### （1）图片的一小关键部分当作输入，不用整张图片当作输入

抓住重要的特征，不用每个neuron都看完整的图片

<img src="图片/3/4.jpg" style="zoom:50%;" />

 **receptive field**

每个neural只关注自己的receptive field，可以有重叠范围，也可以是相同范围

每个field大小可以不同，可以不包含所有channel，一个field可以不相连

如图，这个neuron的输入是一个27维的向量，这个neuron会给这个输入的每个dimension一个weight，再加上bias得到输出，这个输出再送给下一层的neuron作为输入

<img src="图片/3/6.jpg" style="zoom:50%;" />

**一般来说field包含全部channel，所以直说长和宽（长和宽合称kernel size）**，比如3x3（一般都这么大）；但是也可以只考虑一个channel

每个neuron的field的大小可以不一样，可以不相连

**往往一个field，会有一组neural来使用**（比如64、128个）

<img src="图片/3/7.jpg" style="zoom:50%;" />

**strike**

 通过平移一个field，移动后与移动前的**有大量重叠**，得到第二个field，移动的长度就是strike

**overlap**

 超出图片范围

**padding 补零**

 如果移动后，超出图片范围，就补零，也可以补别的数字

所有receptive fields 覆盖整个图片

<img src="图片/3/8.jpg" style="zoom:50%;" />



##### （2）不同receptive field的neural共享参数

**parameter sharing** 共享参数

相同类型内容出现在两张图片的不同field里，这两个field对应的neural都需要加一个侦测这个类型的内容吗？

<img src="图片/3/9.jpg" style="zoom:50%;" />

<img src="图片/3/10.jpg" style="zoom:50%;" />

**不同field**的neuron共享参数，即这两个neuron的**weight是一样**的，只是**输入x不一样**

注意：两个守备相同field的neural不应该共享参数，那么他们输出就都一样了

<img src="图片/3/11.jpg" style="zoom:50%;" />



**filter**

已知每个field有一组neural守备，下面两个field，左上field的第一个neural和右下field的第一个neural共享参数，**这组参数叫filter1**；

左2neural和右2neural共享参数...  所以每个field都只有1组参数

<img src="图片/3/12.jpg" style="zoom:50%;" />

**convolution layer**

也就是receptive field 加上 parameter sharing

通过以上两个方法，从fully connected layer变成convolution layer，**会具有larger model bias**，但是更适合影像辨识

<img src="图片/3/13.jpg" style="zoom:50%;" />

### CNN的第二种介绍方式

**filter**

每个convolution layer里有很多filter

每个filter的作用是去图片里抓取pattern，每个filter的大小是3x3xchannel（channel可以是3或1）

<img src="图片/3/14.jpg" style="zoom:50%;" />

假设channel为1，假设filter里的参数parameter都是已知的

filter和第一个pattern内积inner product（矩阵相乘，每个相乘后相加）

<img src="图片/3/15.jpg" style="zoom:50%;" />

**strike**

表示移动的距离（这里，左右移动是1，上下移动也是1）

<img src="图片/3/16.jpg" style="zoom:50%;" />

<img src="图片/3/17.jpg" style="zoom:50%;" />

filter对角线有3个1，如果pattern也是对角线3个1，那么此时值最大，视为找到了想要的pattern

<img src="图片/3/18.jpg" style="zoom:50%;" />

使用第二个filter，扫过整个图片，得到第二组数字

<img src="图片/3/19.jpg" style="zoom:50%;" />

**feature map**

64个filter，经过计算后得到64组数字，这群数字合称为feature map

<img src="图片/3/20.jpg" style="zoom:50%;" />

feature map可以看成一张新的图片，它的channel就是filter的数量

把这个作为新的输入，丢入第二层，第二层也有filter，这些filter的大小是3x3x64（要和这一层输入的channel符合）

<img src="图片/3/21.jpg" style="zoom:50%;" />

network叠得越深，同样大小的filter，在原来的图片上看的范围越大，可以侦测到大的pattern

<img src="图片/3/22.jpg" style="zoom:50%;" />

第一个版本里说到的neuron共用的参数，就是第二个版本里的filter 

红色线的那个weight就是右边画红圈的数字

filter也是有bias的，只是这里没有提到

<img src="图片/3/23.jpg" style="zoom:50%;" />

不同field的不同neuron 共享参数share parameter实际上就是第二个版本里的filter扫过一张图片的过程

<img src="图片/3/24.jpg" style="zoom:50%;" />

<img src="图片/3/25.jpg" style="zoom:50%;" />

### Pooling

对一张较大的图片做subsampling，举例来说就是把图片的偶数的column，奇数的row都拿掉，图片变为原来的1/4，但是还是能分辨是什么东西

<img src="图片/3/26.jpg" style="zoom:50%;" />

pooling没有参数，不用根据data学习什么东西，只是一个操作operator

### MAX pooling

pooling的某一个版本

举例，如图，通过filter扫图生成一组数字，把这些数字分成2x2（也可以是任何大小）的组别，每个组别选一个代表数字，这里max pooling选的是最大的那个数字

<img src="图片/3/27.jpg" style="zoom:50%;" />

<img src="图片/3/28.jpg" alt="28" style="zoom:50%;" />

这里经过一层convolution的4x4的图片经过pooling，变成2x2的图片

pooling最主要的作用是减少计算量

<img src="图片/3/29.jpg" style="zoom:50%;" />

**flatten**

把得到的矩阵拉直成一个向量，

再把这个向量放进fully connected 的layers里，再经过softmax得到影像辨识的结果

<img src="图片/3/30.jpg" style="zoom:50%;" />

### CNN的应用

下围棋，把围棋看成19x19的图片

分类也是19x19种，告诉你下一个棋子应该下在哪一个位置

<img src="图片/3/31.jpg" style="zoom:50%;" />

<img src="图片/3/32.jpg" style="zoom:50%;" />

<img src="图片/3/33.jpg" style="zoom:50%;" />

CNN无法处理影像放大、缩小的问题

<img src="图片/3/34.jpg" style="zoom:50%;" />
