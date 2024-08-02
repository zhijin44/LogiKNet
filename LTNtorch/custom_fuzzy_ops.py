import torch
from ltn.fuzzy_ops import ConnectiveOperator, pi_0


class MultiConnectiveOperator(ConnectiveOperator):
    """
    Abstract class for multi-operand connective operators.

    This class extends `ConnectiveOperator` to allow multi-operand operations.

    Every multi-operand connective operator implemented in LTNtorch must inherit from this class
    and implement the `__call__()` method for multi-operand logic.
    """
    def __call__(self, *args):
        """
        Implements the behavior of the multi-operand connective operator.

        Parameters
        ----------
        args : :obj:`tuple` of :class:`torch.Tensor`
            Operands on which the connective operator has to be applied.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class AndProd(MultiConnectiveOperator):
    """
    Goguen fuzzy conjunction operator (product operator) for multiple operands.

    :math:`\land_{Goguen}(x_1, x_2, ..., x_n) = x_1 \cdot x_2 \cdot ... \cdot x_n`

    Parameters
    ----------
    stable : :obj:`bool`, default=True
        Flag indicating whether to use the :ref:`stable version <stable>` of the operator or not.

    Attributes
    ----------
    stable : :obj:`bool`
        See `stable` parameter.

    Notes
    -----
    The Goguen fuzzy conjunction could have vanishing gradients if not used in its :ref:`stable <stable>` version.

    Examples
    --------
    >>> import custom_fuzzy_ops as cfo
    >>> import torch
    >>> And = cfo.AndProd()
    >>> p = ltn.Predicate(func=lambda x: torch.nn.Sigmoid()(
    ...                                     torch.sum(x, dim=1)
    ...                                  ))
    >>> x = ltn.Variable('x', torch.tensor([[0.56], [0.9]]))
    >>> y = ltn.Variable('y', torch.tensor([[0.7], [0.2], [0.1]]))
    >>> z = ltn.Variable('z', torch.tensor([[0.8], [0.5]]))
    >>> print(p(x).value)
    tensor([0.6365, 0.7109])
    >>> print(p(y).value)
    tensor([0.6682, 0.5498, 0.5250])
    >>> print(p(z).value)
    tensor([0.6898, 0.6225])
    >>> print(And([p(x), p(y), p(z)]).value)
    tensor([[0.2993, 0.2461, 0.2345],
            [0.3273, 0.2695, 0.2569]])

    .. automethod:: __call__
    """
    def __init__(self, stable=True):
        self.stable = stable

    def __repr__(self):
        return "AndProd(stable=" + str(self.stable) + ")"

    def __call__(self, operands, stable=None):
        """
        Applies the Goguen fuzzy conjunction operator to the given list of operands.

        Parameters
        ----------
        operands : :obj:`list` of :class:`torch.Tensor`
            List of operands on which the operator has to be applied.
        stable : :obj:`bool`, default=None
            Flag indicating whether to use the :ref:`stable version <stable>` of the operator or not.

        Returns
        -----------
        :class:`torch.Tensor`
            The Goguen fuzzy conjunction of the operands.
        """
        stable = self.stable if stable is None else stable
        if stable:
            operands = [pi_0(op) for op in operands]

        result = operands[0]
        for operand in operands[1:]:
            result = torch.mul(result, operand)
        return result
