---
title:  "Map & Filter functions"
date: 2020-05-12
categories: [Tutorial]
tags: [python]
excerpt: "Learn about Map & Filter functions"
author_profile: true
mathjax: true
---

Let's talk about two more built in functions and their use with the [Lambda Expression](https://muzammil-iftikhar.github.io/tutorial/Lambda/). They are **Map** and **Filter** functions.

### Map function

Map function will map a given function one by one over the entire list or any other iteratable.  
Assume you want to get the square of each digit in your list

```python
#First we define the square function
def square(x):
    return x**2
```

```python
#I want to square each element of this list
my_list = [1,10,20,30]
```

```python
#Normal approach without the map function
for n in my_list:
    print(square(n))
1
100
400
900
```

```python
#Squaring the list with map function
# map(<funtion_name>,<your_list>)
print(map(square,my_list))
<map object at 0x000001CF62AD7160>
```

Running it like that will give you a map object. We will have to cast it to list to get our desired results

```python
list(map(square,my_list))
[1, 100, 400, 900]
```

This way our 'for loop' above was squeezed in a single line of code. But that square function is still multiple lines of code. Now using the power of [Lambda](https://muzammil-iftikhar.github.io/tutorial/Lambda/) with map we can squeeze those four lines of code in a single line

```python
list(map(lambda n:n**2, my_list))
[1, 100, 400, 900]
```

### Filter function

Filter, as the name suggests, will filter any iteratable (like list) based on True/False return of a function.  
Let's say you want to filter the even numbers from a list. So we will get a function 'even_filter' that will return True only when even number is passed to it and we will then pass this function to filter along with our list. And filter will return only the elements from our list, where even_filter holds True

```python
#My sample list
my_list = [1,2,3,4,5,6,7,8,9,10]
```

```python
#Now get a function that will return True when even and False when odd
def filter_even(x):
    return x%2==0
```

Now apply that function on the entire list to filter only the even numbers out of it

```python
print(filter(filter_even,my_list))
<filter object at 0x000001CF62B39BE0>
```

Like map, it will also return a filter object. You would have to cast it to a list to get desirable results

```python
list(filter(filter_even,my_list))
[2, 4, 6, 8, 10]
```

Now implementing the entire above code in a single line with Lambda


```python
list(filter(lambda x:x%2==0,my_list))
[2, 4, 6, 8, 10]
```

As a rule of thumb and for memorizing purposes, apply filter on functions that return only True/False like our above 'filter_even' function.

So, **map** will map a function one by one on any iteratable(like list) and **filter** will map the function one by one on any iteratable and it will filter only the ones where True condition holds
