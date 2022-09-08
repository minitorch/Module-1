from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """
    `ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


class Scalar:
    """
    A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(
        self,
        v: float,
        back: ScalarHistory = ScalarHistory(),
        name: Optional[str] = None,
    ):
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        return "Scalar(%f)" % self.data

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> Scalar:
        # TODO: Implement for Task 1.2.
        return Add.apply(self, b)

    def __bool__(self) -> bool:
        return bool(self.data)

    def __lt__(self, b: ScalarLike) -> Scalar:
        # TODO: Implement for Task 1.2.
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        # TODO: Implement for Task 1.2.
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:  # type: ignore[override]
        # TODO: Implement for Task 1.2.
        return EQ.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        # TODO: Implement for Task 1.2.
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        # TODO: Implement for Task 1.2.
        return Neg.apply(self)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    def log(self) -> Scalar:
        # TODO: Implement for Task 1.2.
        return Log.apply(self)

    def exp(self) -> Scalar:
        # TODO: Implement for Task 1.2.
        print(type(Exp.apply(self)))
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        # TODO: Implement for Task 1.2.
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        # TODO: Implement for Task 1.2.
        return ReLU.apply(self)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x: value to be accumulated
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        "True if this variable created by the user (no `last_fn`)"
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None
        # TODO: Implement for Task 1.3.

        # I'm not sure exactly why, but this works.
        # I think the idea is: do backward_pass on the next function down using the global derivative and the context
        # Then, we combine these with the inputs to give us input derivative pairs.
        

        local_derivs = h.last_fn.backward(h.ctx, d_output)
        derivs_vars = list(zip(h.inputs, local_derivs))
        return derivs_vars

        

    def backward(self, d_output: Optional[float] = None) -> None:
        """
        Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    def backprop_step(self, d_output):
        """
        Function made by Ole! Not sure if this is where it should go.

        Performs a single step of backpropagation by calling chain rule, loop through the input deriv pairs,
        add the derivative values or 0.0 if constant, and then return the output list converted to a tuple.
        

        CHANGE: I'm only going to call the chain rule here, and loop through the scalars etc. in backpropagate.
        """
        derivs_vars = self.history.last_fn.chain_rule(d_output)
        deriv_vars_out = []
        for (deriv, var) in derivs_vars:
            if var.is_constant:
                deriv_vars_out.append((0, var))
            else:
                deriv_vars_out.append((deriv, var))
        return deriv_vars_out



def derivative_check(f: Any, *scalars: Scalar) -> None:
    """
    Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Parameters:
        f : function from n-scalars to 1-scalar.
        *scalars  : n input scalar values.
    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
