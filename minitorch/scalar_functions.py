from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        raise NotImplementedError("Need to implement for Task 1.2")

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # TODO: Implement for Task 1.4.
        raise NotImplementedError("Need to implement for Task 1.4")


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        raise NotImplementedError("Need to implement for Task 1.2")

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        raise NotImplementedError("Need to implement for Task 1.4")


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        raise NotImplementedError("Need to implement for Task 1.2")

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        raise NotImplementedError("Need to implement for Task 1.4")


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        raise NotImplementedError("Need to implement for Task 1.2")

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        raise NotImplementedError("Need to implement for Task 1.4")


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        raise NotImplementedError("Need to implement for Task 1.2")

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        raise NotImplementedError("Need to implement for Task 1.4")


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        raise NotImplementedError("Need to implement for Task 1.2")

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        raise NotImplementedError("Need to implement for Task 1.4")


class LT(ScalarFunction):
    "Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        raise NotImplementedError("Need to implement for Task 1.2")

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # TODO: Implement for Task 1.4.
        raise NotImplementedError("Need to implement for Task 1.4")


class EQ(ScalarFunction):
    "Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        raise NotImplementedError("Need to implement for Task 1.2")

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # TODO: Implement for Task 1.4.
        raise NotImplementedError("Need to implement for Task 1.4")


# class FunctionBase(Generic[T]):
#     """
#     A function that can act on :class:`Variable` arguments to
#     produce a :class:`Variable` output, while tracking the internal history.

#     Call by :func:`FunctionBase.apply`.
#     """

#     @staticmethod
#     def variable(raw, history: Optional[History]):
#         # Implement by children class.
#         raise NotImplementedError()

#     @classmethod
#     def apply(cls, *vals):
#         """
#         Apply is called by the user to run the Function.
#         Internally it does three things:

#         a) Creates a Context for the function call.
#         b) Calls forward to run the function.
#         c) Attaches the Context to the History of the new variable.

#         There is a bit of internal complexity in our implementation
#         to handle both scalars and tensors.

#         Args:
#             vals (list of Variables or constants) : The arguments to forward

#         Returns:
#             `Variable` : The new variable produced

#         """
#         # Go through the variables to see if any needs grad.
#         raw_vals = []
#         need_grad = False
#         for v in vals:
#             if isinstance(v, Variable):
#                 if v.history is not None:
#                     need_grad = True
#                 v.used += 1
#                 raw_vals.append(v.get_data())
#             else:
#                 raw_vals.append(v)

#         # Create the context.
#         ctx = Context(not need_grad)

#         # Call forward with the variables.
#         c = cls.forward(ctx, *raw_vals)
#         assert isinstance(c, cls.data_type), "Expected return typ %s got %s" % (
#             cls.data_type,
#             type(c),
#         )

#         # Create a new variable from the result with a new history.
#         back = None
#         if need_grad:
#             back = History(cls, ctx, vals)
#         return cls.variable(cls.data(c), back)

#     @classmethod
#     def chain_rule(cls, ctx: Context,
#                    inputs: Union[float, Sequence[float]],
#                    d_output: float) -> Sequence[Tuple[Variable, float]]:
#         """
#         Implement the derivative chain-rule.

#         Args:
#             ctx (:class:`Context`) : The context from running forward
#             inputs (list of args) : The args that were passed to :func:`FunctionBase.apply` (e.g. $x, y$)
#             d_output (number) : The `d_output` value in the chain rule.

#         Returns:
#             list of (`Variable`, number) : A list of non-constant variables with their derivatives
#             (see `is_constant` to remove unneeded variables)

#         """
#         # Tip: Note when implementing this function that
#         # cls.backward may return either a value or a tuple.
#         # ASSIGN1.3
#         d_inputs = cls.backward(ctx, d_output)
#         d_inputs = wrap_tuple(d_inputs)
#         return [
#             (inp, inp.expand(d_input))
#             for inp, d_input in zip(inputs, d_inputs)
#             if not is_constant(inp)
#         ]
#         # END ASSIGN1.3
