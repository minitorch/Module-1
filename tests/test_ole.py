from typing import Tuple

import pytest

import minitorch
from minitorch import Context, ScalarFunction, ScalarHistory

# ## Task 1.3 - Tests for the autodifferentiation machinery.

# Simple sanity check and debugging tests.


class Function1(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        "$f(x, y) = x + y + 10$"
        return x + y + 10
                
    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        "Derivatives are $f'_x(x, y) = 1$ and $f'_y(x, y) = 1$"
        return d_output, d_output




def test_topological_sort():
    x = minitorch.Scalar(2)
    y = minitorch.Scalar(3)
    z = (x * x) * y + 10.0 * x
    z.backward(d_output=1)
    return minitorch.autodiff.topological_sort(z)

#L = test_topological_sort()

#print("L is" + str(L))


class Function2(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        "$f(x, y) = x \times y + x$"
        ctx.save_for_backward(x, y)
        return x * y + x

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        "Derivatives are $f'_x(x, y) = y + 1$ and $f'_y(x, y) = x$"
        x, y = ctx.saved_values
        return d_output * (y + 1), d_output * x



def test_backprop1() -> None:
    # Example 1: F1(0, v)
    var = minitorch.Scalar(0.1)
    var_test = minitorch.Scalar(0)
    var2 = Function1.apply(0.1, var)
    var2.backward(d_output=5)
    print(var.derivative, var_test.derivative)

test_backprop1()