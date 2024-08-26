import math
from typing import Callable, Iterable, Tuple
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 2e-5,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        l1_weight_decay: float = 0.001,
        l2_weight_decay: float = 0.01,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            l1_weight_decay=l1_weight_decay,
            l2_weight_decay=l2_weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Apply bias correction
                if group["correct_bias"]:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = group["lr"]

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # Apply L1 regularization
                p.data.sign_().mul_(torch.max(torch.abs(p.data) - group["l1_weight_decay"] * step_size, torch.zeros_like(p.data)))

                # Apply L2 regularization
                if group["weight_decay"] != 0 or group["l2_weight_decay"] != 0:
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    p.data.add_(p.data, alpha=-group["lr"] * (group["weight_decay"] + group["l2_weight_decay"]))

        return loss