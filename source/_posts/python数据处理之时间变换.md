---
title: python数据处理之时间变换
date: 2020-09-19 10:21:34
categories: python学习 
tags: 编程技能， 时序数据
toc: true 
mathjax: true 
---
本文将就`python`中对时间标识的处理做一点讲解，在进行数据分析时，有些时候我们需要对时间序列数据进行处理，比如像下面这样一组数据<!--more-->:

| Time | feature1 | ... | feature n |
| :-----:| :----: | :----: |:----:|
| 2013年-01月-01日: 00时00分00秒 | ... | ... |  ...| 
| 2013年-01月-01日: 02时00分00秒 | ... | ... |    ...  | 
| ... | ... | ... |    ...  | 
| 2015年-01月-01日: 00时00分00秒 | ... | ... |    ...  | 

对于这样一段时序数据，采样间隔2小时。当我们拿到这组数据之后，首先要考虑的问题就是这组数据有没有缺失，而判断数据记录有无缺失的关键在于其时间序列`Time`是否连续。我们一般采用`pandas.read_csv()`函数来进行`csv`文件中数据的读取，读取得到的第一列是一个字符串形式，通过这些字符串来判断时间是否连续是比较困难的，需要重新写一个函数来对字符串进行处理。但幸运的是，其实已经有函数可以帮我们完成这样一项工作了，下面介绍下本文的关键对象---`unix时间戳`：
> **unix时间戳:** unix时间戳是一种跟踪时间（以秒为单位）的方式。此计数从1970年1月1日UTC的Unix Epoch开始。因此，unix时间戳仅仅是特定日期与Unix纪元之间的秒数。

简而言之，unix时间戳就是一个`int`型的数据，其值代表了某个时间点距离`1997-01-01: 00:00:00`的秒数，那么如果我们能够将我们的时间字符串转换成unix时间戳，那么后面的数据拆分、数据缺失定位就更加得心应手了。本文按照以下结构进行组织:
- 常见日期时间类型 
- 日期格式转换 
- 总结

### 常见日期时间类型 
在进行时间序列数据处理时，常见的日期时间类型大概有以下四种:
 
| 类型| 格式 |  示例 |
| :-----| :---- |:----:|
| time | 时间格式 |  17:00:00| 
| date | 日期格式 | 2019-09-19  | 
| datetime | 日期时间格式 |  2019-09-19: 17:00:00  | 
| timestamp | 时间戳格式 |  1568912400 | 

实际上以上四种类型可以粗分为两个类型: 字符串类型和时间戳格式，时间字符串直接转时间戳格式并不自然，因此在python中实现两者转化还需借助一个中间变量类型`struct_time`，时间格式变换流程如下:
$$
    str \Leftrightarrow struct \, time \Leftrightarrow stamp 
$$
字符串格式和时间戳格式通过我上面的讲解大家应该可以有一个基本认识，下面重点介绍下`struct_time`类型，它是一个具有命名元祖接口的对象:可以通过索引和属性名访问值，存在以下值:

| index | attribute | values |
| :-----:| :----: |:----:| 
|    0   |    tm-year   | 年份  | 
|    1   |    tm-mon   | 月份range(1,12)  | 
|    2   |    tm-mday   |  天数range(1,31) | 
|    3   |    tm-hour   | 小时range(0,23)  | 
|    4   |    tm-min   | 分钟range(0,59)  | 
|    5   |    tm-sec   | 秒数range(0,59)  | 
|    6   |    tm-wday   | 星期range(0,6)  | 
|    7   |    tm-yday   | 年中的一天range(1,366)  | 
而要实现这样的变换，则就要提到`python`中两个专门与时间打交道的模块`time`和`datetime`。

### 日期和格式转换
这部份将介绍以下`time`以及`datetime`中常用的一些方法。
#### `time`模块
[`time`](https://docs.python.org/3/library/time.html)模块是操作系统层面上的， 与`os`模块类似，主要提供一些系统层面的与时间有关的操作，该模块是其他所有与时间相关模块的基础，我们用到该模块比较多的方法主要有以下几个:
- `time.time()`: 返回当前时间的时间戳 
```python
>>>: 
print(time.time())
>>>: 
1600502127.7830698
```

- `time.loacltime()`: 将一个`unix stamp`转换为本地时间(东八区)的`struct_time`类型 
```python
>>>: 
print(time.localtime(time.time())) 
>>>: 
time.struct_time(tm_year=2020, tm_mon=9, tm_mday=19, tm_hour=16, tm_min=2, tm_sec=55, tm_wday=5, tm_yday=263, tm_isdst=0)
```
- `time.gmtime()`: 将一个`unix stamp`转换为格林威治时间的`struct_time`类型 
```python
>>>: 
print(time.gmtime(time.time()))
>>>: 
time.struct_time(tm_year=2020, tm_mon=9, tm_mday=19, tm_hour=8, tm_min=5, tm_sec=31, tm_wday=5, tm_yday=263, tm_isdst=0)
```
- `time.strftime(format, struct_time)`: 将一个`struct_time`类型输出转换为指定格式的字符串  
```python
>>>: 
format = '%Y-%m-%d: %H:%M:%S' 
struct_time = time.localtime(time.time()) 
print(time.strftime(format, struct_time)) 
>>>: 
2020-09-19: 16:10:42 
```
- `time.strptime(str, format)`: 与`strftime`作用相反，将一个时间字符串按照`format`格式解析成`struct_time`类型 
```python 
>>>: 
format = '%Y-%m-%d: %H:%M:%S'  
time_str = '2020-9-19: 16:17:00' 
print(time.strptime(time_str, format))  
>>>: 
time.struct_time(tm_year=2020, tm_mon=9, tm_mday=19, tm_hour=16, tm_min=17, tm_sec=0, tm_wday=5, tm_yday=263, tm_isdst=-1)
```

- `time.mktime(struct_time)`: 将一个`struct_time`类型数据转化成`unxi_stample`, 该操作是`time.localtime()`的反操作，因此其默认输入的`struct_time`是本地时间
``` python 
>>>: 
format = '%Y-%m-%d: %H:%M:%S'  
time_str = '2020-9-19: 16:17:00' 
time_struct = time.strptime(time_str, format) 
print(time_struct)
print(time.mktime(time_struct)) 
print(time.localtime(time.mktime(time_struct)))
>>>:
time.struct_time(tm_year=2020, tm_mon=9, tm_mday=19, tm_hour=16, tm_min=17, tm_sec=0, tm_wday=5, tm_yday=263, tm_isdst=-1)
1600503420.0
time.struct_time(tm_year=2020, tm_mon=9, tm_mday=19, tm_hour=16, tm_min=17, tm_sec=0, tm_wday=5, tm_yday=263, tm_isdst=0)
```

上面这些方法实际上就已经完全可以实现我们的功能，大致是下面这样一张图:

![转换方法](https://raw.githubusercontent.com/xuejy19/xuejy19.github.io/source/Img/time.png)

#### `datetime`模块 
`datetime`模块可以看作是对`time`模块的一个封装和扩展，上面提到的`time`中常用方法在`datetime`中均有对应, 在`datetime`下主要有以下几个类：
- `date`: 处理与日期相关的数据，属性: year, month, day 
- `time`: 处理与某天时间相对应的数据， 属性: hour, minute, second, microsecond, tzinfo 
- `datetime`: 日期和时间的组合，属性: year, month, day, hour, minute, second, microsecond, tzinfo 
- `timedelta`: 与持续时间计算有关  
- `tzinfo`: 时区信息类  

`datetime`类基本已经可以实现上面`time`模块所能实现的功能，具体对应大家可以去官网[datetime](https://docs.python.org/2/library/datetime.html)查看，我后面如果需要用到也会进行总结。 

#### `pandas.to_datetime()` 
我们读取数据往往需要使用`pandas`库，得到`Dateframe`类型的数据，读取之后往往某一列就是时间序列，若用前面的方法来做转换，需要对每一个时间字符串操作，比较麻烦。为了解决该问题，就需要使用`pandas.to_datetime()`方法，使用该方法可以将变量变为`datetime`类型，方便后面进行操作。
- 字符串时间转时间戳
```python
data.time = pandas.to_datetime(data.time) 
data.time = data.time.apply(lambda x: time.mktime(x.tuple))
```
- 时间戳转字符串 
```python
data.time = pandas.to_datetime(data.time) 
data.time = data.time.apply(lambda x: time.strftime(format, x.tuple))
```

### 总结 
在进行数据处理的时候我们往往不能够默认数据是“好数据”，因此就必须通过程序来对数据进行验证，而对于时序数据，时间信息的处理是绕不过去的，掌握了`python`中常用的对时间处理方法，后面处理时序数据可以更加得心应手。