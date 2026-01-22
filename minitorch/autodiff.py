from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_pos = list(vals)
    vals_neg = list(vals)

    vals_pos[arg] = vals[arg] + epsilon
    vals_neg[arg] = vals[arg] - epsilon

    f_pos = f(*vals_pos)
    f_neg = f(*vals_neg)

    return (f_pos - f_neg) / (2.0 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    post: List[Variable] = []

    def dfs(v: Variable) -> None:
        uid = v.unique_id
        if uid in visited:
            return
        visited.add(uid)
        if v.is_constant():
            return
        for p in v.parents:
            dfs(p)
        post.append(v)

    dfs(variable)
    return list(reversed(post))


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    topo = list(topological_sort(variable))

    grads: dict[int, float] = {}
    grads[variable.unique_id] = float(deriv)

    for v in topo:
        g_out = float(grads.get(v.unique_id, 0.0))
        if v.is_leaf():
            v.accumulate_derivative(g_out)
            continue

        for parent, g_local in v.chain_rule(g_out):
            pid = parent.unique_id
            grads[pid] = grads.get(pid, 0.0) + float(g_local)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
