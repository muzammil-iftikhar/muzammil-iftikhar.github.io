---
title:  "Args and Kwargs"
date: 2020-06-01
categories: [Tutorial]
tags: [python,args,kwargs]
excerpt: "Learn about the usage of *args and **kwargs keywords"
author_profile: true
mathjax: true
---

Lets learn about `*args` and `**kwargs` keywords that show up alot as parameters in functions.

```python
def add_func(a,b):
    print(a+b)    
```

```python
add_func(2,3)

    5
```

Our function can take only two input parameters `a` and `b`.

### *args

To make a function so that it accepts, practically unlimited, number of input paramters, we use the `*args` keyword. It will build the tuple out of your input parameters.

```python
def add_func(*args):
    print(sum(args))
```

```python
add_func(2,3,4,5,6,7,8)

    35
```

### **kwargs

It will build a dictionary of key:value pairs based on your input.

```python
def func(**kwargs):
    print(kwargs)
```

```python
func(name='muzammil',fruit='apple',balls=3)

    {'name': 'muzammil', 'fruit': 'apple', 'balls': 3}
```

Let's combine both `*args` and `**kwargs`.

```python
def func(*args,**kwargs):
    if 'fruit' in kwargs.keys():
        print(f"I have {sum(args)} number of {kwargs['fruit']}")
    else:
        pass
```

```python
func(2,3,fruit='apple')

    I have 5 number of apple
```
