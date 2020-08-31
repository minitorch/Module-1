import uuid


def wrap_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):
    if len(x) == 1:
        return x[0]
    return x


class Variable:
    """
    Attributes:
        history (:class:`History`) : the Function calls that created this variable or None if constant
        derivative (number): the derivative with respect to this variable
        name (string) : an optional name for debugging
    """

    def __init__(self, history, name=None):
        assert history is None or isinstance(history, History), history

        self.history = history
        self._derivative = None

        # For debugging can have a name.
        if name is not None:
            self.name = name
        else:
            self.name = str(uuid.uuid4())

    def requires_grad_(self, val):
        self.history = History(None, None, None)

    def backward(self, d_output=None):
        """
        Calls autodiff to fill in the derivatives for the history of this object.
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(VariableWithDeriv(self, d_output))

    @property
    def derivative(self):
        return self._derivative

    ## IGNORE
    def __hash__(self):
        return hash(self._name)

    def _add_deriv(self, val):
        assert self.history.is_leaf(), "Only leaf variables can have derivatives."
        if self._derivative is None:
            self._derivative = self.zeros()
        self._derivative += val

    def zero_grad_(self):
        self._derivative = self.zeros()

    def __radd__(self, b):
        return self + b

    def __rmul__(self, b):
        return self * b

    def zeros(self):
        return 0.0

    def expand(self, x):
        return x

    ## IGNORE


class Context:
    """
    Context class is used by.
    """

    def __init__(self, no_grad=False):
        self._saved_values = None
        self.no_grad = no_grad

    def save_for_backward(self, *values):
        if self.no_grad:
            return
        self._saved_values = values

    @property
    def saved_values(self):
        assert not self.no_grad, "Doesn't require grad"
        assert self._saved_values is not None, "Did you forget to save values?"
        return unwrap_tuple(self._saved_values)


class History:
    """
    `History` stores all of the `Function` operations that were used to
    construct an autodiff object.

    Attributes:
        last_fn (:class:`FunctionBase`) : The last function that was called.
        ctx (:class:`Context`): The context for that function.
        inputs (list of inputs) : The inputs that were given when `last_fn.forward` was called.
    """

    def __init__(self, last_fn=None, ctx=None, inputs=None):
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs

    def is_leaf(self):
        return self.last_fn is None

    def backprop_step(self, d_output):
        return self.last_fn.chain_rule(self.ctx, self.inputs, d_output)


class VariableWithDeriv:
    "Holder for a variable with it derivative."

    def __init__(self, variable, deriv):
        self.variable = variable
        self.deriv = variable.expand(deriv)


class FunctionBase:
    """
    A function that can act on :class:`Variable` arguments to
    produce a :class:`Variable` output, while tracking the internal history.

    Call by :func:`FunctionBase.apply`.

    """

    @staticmethod
    def variable(raw, history):
        pass

    @classmethod
    def apply(cls, *vals):
        raw_vals = []
        need_grad = False
        for v in vals:
            if isinstance(v, Variable):
                if v.history is not None:
                    need_grad = True
                raw_vals.append(v.get_data())
            else:
                raw_vals.append(v)
        ctx = Context(not need_grad)
        c = cls.forward(ctx, *raw_vals)
        assert isinstance(c, cls.data_type), "Expected return typ %s got %s" % (
            cls.data_type,
            type(c),
        )
        back = None
        if need_grad:
            back = History(cls, ctx, vals)
        return cls.variable(cls.data(c), back)

    @classmethod
    def chain_rule(cls, ctx, inputs, d_output):
        """
        Implement the derivative chain-rule.

        Args:
            cls (:class:`FunctionBase`): The Function
            ctx (:class:`Context`) : The context from running forward
            inputs (list of args) : The args that were passed to :func:`FunctionBase.apply` (e.g. :math:`x, y`)
            d_output (number) : The `d_output` value in the chain rule.

        Returns:
            list of :class:`VariableWithDeriv`: A list of non-constant variables with their derivatives
            (see `is_constant` to remove unneeded variables)

        """
        # TODO: Implement for Task 1.3.
        raise NotImplementedError('Need to implement for Task 1.3')


def is_leaf(val):
    return isinstance(val, Variable) and val.history.is_leaf()


def is_constant(val):
    return not isinstance(val, Variable) or val.history is None


def backpropagate(final_variable_with_deriv):
    """
    Runs a breadth-first search on the computation graph in order to
    backpropagate derivatives to the leaves.

    See :doc:`backpropagate` for details on the algorithm

    Args:
       final_variable_with_deriv (:class:`VariableWithDeriv`): The final variable
           and its derivative that we want to propagate backward to the leaves.
    """
    # TODO: Implement for Task 1.4.
    raise NotImplementedError('Need to implement for Task 1.4')
