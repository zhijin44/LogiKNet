import torch
from ltn.fuzzy_ops import ConnectiveOperator, pi_0, AndProd as AndProdBinary

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
    """
    def __init__(self, stable=True):
        """
        Constructor to set whether to use the stable version of the Goguen fuzzy conjunction.
        
        Parameters
        ----------
        stable : bool, default=True
            A boolean flag indicating whether to use the stable version of the operator.
        """
        self.stable = stable
        self.binary_conjunction = AndProdBinary(stable)

    def __repr__(self):
        return f"AndProd(stable={self.stable})"

    def __call__(self, *args, stable=None):
        """
        Applies the Goguen fuzzy conjunction operator to a variable number of operands.

        Parameters
        ----------
        *args : :class:`torch.Tensor`
            A variable number of operands on which the operator has to be applied.
        stable : :obj:`bool`, default=None
            Flag indicating whether to use the stable version of the operator or not.

        Returns
        -----------
        :class:`torch.Tensor`
            The Goguen fuzzy conjunction of the operands.
        """

        stable = self.stable if stable is None else stable
        result = args[0]
        for operand in args[1:]:
            result = self.binary_conjunction(result, operand, stable=stable)

        return result
