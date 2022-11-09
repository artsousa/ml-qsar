"""
    Copyright Notice

    Copyright (c) 2020 GlaxoSmithKline LLC (Kim Branson & Cuong Nguyen)
    Copyright (c) 2019 Debajyoti Datta, Ian Bunner, Praateek Mahajan, Sebastien Arnold

    This copyright work was created in 2020 and is based on the 
    copyright work created in 2018 available under the MIT License at 
    https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py

    ------------

    Modified MAML class from learn2learn that support running in ANIL mode
    
    The base maml class and update methods are from - 
        https://github.com/GSK-AI/meta-learning-qsar/blob/master/src/models/l2l_maml.py

    further alterations to get to - will be made
        https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch
        
"""

import traceback

import torch
from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module
from torch.autograd import grad

def maml_update(model, lr, grads=None, anil=False):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)
    **Description**
    Performs a MAML update on model using grads and lr.
    The function re-routes the Python object, thus avoiding in-place
    operations.
    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.
    **Arguments**
    * **model** (Module) - The model to update.
    * **lr** (float) - The learning rate used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the gradients in .grad attributes.
    **Example**
    ~~~python
    maml = l2l.algorithms.MAML(Model(), lr=0.1)
    model = maml.clone() # The next two lines essentially implement model.adapt(loss)
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    maml_update(model, lr=0.1, grads)
    ~~~
    """
    if grads is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            msg = "WARNING:maml_update(): Parameters and gradients have different length. ("
            msg += str(len(params)) + " vs " + str(len(grads)) + ")"
            print(msg)
        for i, (p, g) in enumerate(zip(params, grads)):
            if anil and i < (len(params) - 2):
                g = torch.zeros_like(g)
            p.grad = g

    # Update the params
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - lr * p.grad

    # Second, handle the buffers if necessary
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None:
            model._buffers[buffer_key] = buff - lr * buff.grad

    # Then, recurse for each submodule
    for module_key in model._modules:
        model._modules[module_key] = maml_update(
            model._modules[module_key], lr=lr, grads=None, anil=anil
        )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    model._apply(lambda x: x)
    return model


class MAML(BaseLearner):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)
    **Description**
    High-level implementation of *Model-Agnostic Meta-Learning*.
    This class wraps an arbitrary nn.Module and augments it with `clone()` and `adapt()`
    methods.
    For the first-order version of MAML (i.e. FOMAML), set the `first_order` flag to `True`
    upon initialization.

    **Arguments**
    * **model** (Module) - Module to be wrapped.
    * **lr** (float) - Fast adaptation learning rate.
    * **first_order** (bool, *optional*, default=False) - Whether to use the first-order
        approximation of MAML. (FOMAML)
    * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to `allow_nograd`.
    * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
        parameters that have `requires_grad = False`.
    **References**
    1. Finn et al. 2017. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks."
    **Example**
    ~~~python
    linear = l2l.algorithms.MAML(nn.Linear(20, 10), lr=0.01)
    clone = linear.clone()
    error = loss(clone(X), y)
    clone.adapt(error)
    error = loss(clone(X), y)
    error.backward()
    ~~~
    """

    def __init__(
        self,
        model,
        lr,
        first_order=False,
        allow_unused=None,
        allow_nograd=False,
        anil=False,
    ):
        super(MAML, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused
        self.anil = anil

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(
        self, loss, first_order=None, allow_unused=None, allow_nograd=None, anil=None
    ):
        """
        **Description**
        Takes a gradient step on the loss and updates the cloned parameters in place.
        **Arguments**
        * **loss** (Tensor) - Loss to minimize upon update.
        * **first_order** (bool, *optional*, default=None) - Whether to use first- or
            second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
            of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=None) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.
        """
        if anil is None:
            anil = self.anil
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        if allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = grad(
                loss,
                diff_params,
                retain_graph=second_order,
                create_graph=second_order,
                allow_unused=allow_unused,
            )
            gradients = []
            grad_counter = 0

            # Handles gradients for non-differentiable parameters
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(
                    loss,
                    self.module.parameters(),
                    retain_graph=second_order,
                    create_graph=second_order,
                    allow_unused=allow_unused,
                )
            except RuntimeError:
                traceback.print_exc()
                print(
                    "learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?"
                )

        # Update the module
        self.module = maml_update(self.module, self.lr, gradients, anil)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Description**
        Returns a `MAML`-wrapped copy of the module whose parameters and buffers
        are `torch.clone`d from the original module.
        This implies that back-propagating losses on the cloned module will
        populate the buffers of the original module.
        For more information, refer to learn2learn.clone_module().
        **Arguments**
        * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
            or second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(
            clone_module(self.module),
            lr=self.lr,
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
        )
