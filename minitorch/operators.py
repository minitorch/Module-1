import math

## Task 0.1
## Mathematical operators


def mul(x, y):
    ":math:`f(x, y) = x * y`"
    raise NotImplementedError('Need to include this file from past assignment.')


def id(x):
    ":math:`f(x) = x`"
    raise NotImplementedError('Need to include this file from past assignment.')


def add(x, y):
    ":math:`f(x, y) = x + y`"
    raise NotImplementedError('Need to include this file from past assignment.')


def neg(x):
    ":math:`f(x) = -x`"
    raise NotImplementedError('Need to include this file from past assignment.')


def lt(x, y):
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    raise NotImplementedError('Need to include this file from past assignment.')


def eq(x, y):
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    raise NotImplementedError('Need to include this file from past assignment.')


def max(x, y):
    ":math:`f(x) =` x if x is greater than y else y"
    raise NotImplementedError('Need to include this file from past assignment.')


def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    """
    raise NotImplementedError('Need to include this file from past assignment.')


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)
    """
    raise NotImplementedError('Need to include this file from past assignment.')


def relu_back(x, y):
    ":math:`f(x) =` y if x is greater than 0 else 0"
    raise NotImplementedError('Need to include this file from past assignment.')


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(a, b):
    return b / (a + EPS)


def inv(x):
    ":math:`f(x) = 1/x`"
    return 1.0 / x


def inv_back(a, b):
    return -(1.0 / a ** 2) * b


## Task 0.3
## Higher-order functions.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    raise NotImplementedError('Need to include this file from past assignment.')


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) one each pair of elements.

    """
    raise NotImplementedError('Need to include this file from past assignment.')


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`

    """
    raise NotImplementedError('Need to include this file from past assignment.')


def sum(ls):
    """
    Sum up a list using :func:`reduce` and :func:`add`.
    """
    raise NotImplementedError('Need to include this file from past assignment.')


def prod(ls):
    """
    Product of a list using :func:`reduce` and :func:`mul`.
    """
    raise NotImplementedError('Need to include this file from past assignment.')
