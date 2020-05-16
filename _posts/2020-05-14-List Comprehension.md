---
title:  "List Comprehensions"
date: 2020-05-14
categories: [Tutorial]
tags: [python]
excerpt: "Learn about List comprehensions"
author_profile: true
mathjax: true
---

Another awesome python feature is **List Comprehension**. It is basically a one liner for loop built inside brackets

```python
lst = [x for x in range(10)]
```

```python
print(lst)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
``` 

You can even run mathematical functions inside list comprehension

```python
lst = [x**2 for x in range(10)]
```

```python
print(lst)
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

Making a list out of any word

```python
lst = [x for x in 'hello']
```

```python
print(lst)
['h', 'e', 'l', 'l', 'o']
```

### List comprehension with only 'If' statement

```python
lst = [x for x in range(10) if x%2==0]
```

```python
print(lst)
[0, 2, 4, 6, 8]
```

### List comprehension with 'If-else' statement

```python
lst = [x if x%2==0 else 'odd' for x in range(10)]
```

```python
print(lst)
[0, 'odd', 2, 'odd', 4, 'odd', 6, 'odd', 8, 'odd']
```

### Nested list comprehensions

```python
lst = [x*2 for x in [x*2 for x in range(5)]]
```

```python
print(lst)
[0, 4, 8, 12, 16]
```

### Nested for loops inside list comprehensions

```python
[(x,y) for x in range(3) for y in range(3)]
[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
```

```python
for x in range(3):
    for y in range(3):
        print(f'({x},{y})',end=', ')
(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
```
