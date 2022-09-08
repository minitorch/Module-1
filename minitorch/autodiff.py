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
    # TODO: Implement for Task 1.1.
    # Make the first set of arguments
    upper = list(vals[:])
    upper[arg] = upper[arg] + epsilon/2

    # Then the second
    lower = list(vals[:])
    lower[arg] = lower[arg] - epsilon/2

    # Return the central difference. Note the * here unpacks the list into the arguments of the function
    return (f(*upper) - f(*lower)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    # I think these have all been implemented somewhere??
    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass



def find_nodes(variable):
    """
    Helper function to find all descendent modules of a module.
    """

    # TODO: PROBLEM: variables are not a class of modules. Hence, we need to implement the below stuff in terms of scalars instead.
    # Have maybe solved this, but it needs checking
    lst = [variable]    
    # find the direct descendents of the node, add these to the list
    children = variable.parents
    lst += children
    # if there are none, return the list
    if children == []:
        return lst
    # otherwise, find the nodes of all descendent modules, and append these to the list. Recursive.
    else:
        for child in children:
            lst += find_nodes(child)
    return lst

def visit(node_n, L):
    """
    Helper function for topological sort.

    Outlines the visit function found in the depth first search pseudocode here:
    https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
    """
    # If node already added, stop the visit
    if node_n in L:
        return L
    # Otherwise, visit each child node
    children = node_n.parents
    for child in children:
        # update the list to reflect what happens after the search
        L = visit(child, L)
    #prepend the list with your original node
    # Unsure how to format this: should L be composed of names? or unique ids?
    L.insert(0, node_n)
    return L



def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Works by Depth-first search.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.

    # TODO: Also general testing

    # To implement topological sort, we first need to form a list of all nodes, from the first node.
    nodes = find_nodes(variable)

    # Empty list which will contain the sorted nodes
    L = []

    while nodes != []:
        #print(nodes)
        node = nodes[0]
        L = visit(node, L)
        #print(L)
        nodes = nodes[1:]
    return L


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # Form an ordered queue
    queue = topological_sort(variable)
    # To help us use unique_id, we are going to create a dictionary of modules with their unique_ids
    queue_ids = [x.unique_id for x in queue]
    queue_with_ids = dict(zip(queue_ids, queue))

    # Create dictionary of scalars and current derivatives
    current_backprop = dict()
    current_backprop[variable.unique_id] = deriv

    for node in queue:
        # store derivative
        deriv = current_backprop[node.unique_id]
        # If the node is a leaf, then change its derivative.
        if node.is_leaf():
            #print(node.is_leaf())
            node.accumulate_derivative(deriv)
            print("node, deriv are" + str(node) + str(deriv))
            print("node derivative (within the list is " + str(node.derivative))
            #assert node.derivative == 5
        # If not, call backprop_step on the node, and add deriv to that scalar's total deriv.
        else:
            next_scalars_derivs = node.backprop_step(deriv)
            for (scalar, deriv2) in next_scalars_derivs:
                print("deriv2 is" + str(deriv2))
                if scalar.unique_id in current_backprop.keys():
                    current_backprop[scalar.unique_id] += deriv2
                else:
                    current_backprop[scalar.unique_id] = deriv2
    print("current_backprop is" + str(current_backprop))

    print("variable.derivative is "+ str(variable.parents[1].derivative))

    #assert variable.parents[0].derivative == 5
    return




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
