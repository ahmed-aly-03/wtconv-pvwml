"""Implements the loss/attack machinery from:
"Decoupling Representation Learning and Classifier for Long-Tailed
Adversarial Training" (TDLF), Pattern Recognition 172 (2026) 112607.

Used for script 2's representation-learning stage:
  - SupervisedContrastiveLoss: Eq. 4
  - four-view construction (clean_i, adv_i, clean_j, adv_j): Eq. 6-8
  - AdversarialWeightPerturbation: Eq. 9-10
"""
import torch
import torch.nn as nn


class SupervisedContrastiveLoss(nn.Module):
    """Eq. 4. `embeddings` must be L2-normalized, shape (N, D). `labels`
    shape (N,) -- positives are all other samples sharing the same label
    (this is what makes it *supervised* contrastive rather than SimCLR's
    same-source-image-only positives)."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        n = embeddings.shape[0]

        sim = torch.matmul(embeddings, embeddings.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # numerical stability

        labels = labels.view(-1, 1)
        same_class = torch.eq(labels, labels.T).float().to(device)
        self_mask = torch.eye(n, device=device)
        pos_mask = same_class * (1 - self_mask)

        exp_sim = torch.exp(sim) * (1 - self_mask)  # A(k) = I \ {k}
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_counts = pos_mask.sum(dim=1).clamp(min=1.0)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_counts

        return -mean_log_prob_pos.mean()


def generate_adversarial_views(
    model: nn.Module,
    scl_criterion: SupervisedContrastiveLoss,
    x_i: torch.Tensor,
    x_j: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    num_steps: int,
):
    """Eq. 6-8: find delta_i, delta_j (bounded by epsilon in L-inf, on raw
    [0,1] pixels) that maximize the supervised contrastive loss computed
    over the 4-view batch {adv_i, adv_j, clean_i, clean_j}, holding model
    weights fixed. Returns the adversarial views (detached)."""
    was_training = model.training
    model.eval()  # freeze dropout/droppath while searching for the attack

    delta_i = (torch.rand_like(x_i) * 2 - 1) * epsilon
    delta_j = (torch.rand_like(x_j) * 2 - 1) * epsilon
    delta_i = (x_i + delta_i).clamp(0, 1) - x_i
    delta_j = (x_j + delta_j).clamp(0, 1) - x_j

    view_labels = labels.repeat(4)

    for _ in range(num_steps):
        delta_i.requires_grad_(True)
        delta_j.requires_grad_(True)

        adv_i = (x_i + delta_i).clamp(0, 1)
        adv_j = (x_j + delta_j).clamp(0, 1)
        views = torch.cat([adv_i, adv_j, x_i, x_j], dim=0)

        embeddings = model(views)
        loss = scl_criterion(embeddings, view_labels)
        grad_i, grad_j = torch.autograd.grad(loss, [delta_i, delta_j])

        with torch.no_grad():
            delta_i = delta_i + step_size * grad_i.sign()
            delta_j = delta_j + step_size * grad_j.sign()
            delta_i = delta_i.clamp(-epsilon, epsilon)
            delta_j = delta_j.clamp(-epsilon, epsilon)
            delta_i = (x_i + delta_i).clamp(0, 1) - x_i
            delta_j = (x_j + delta_j).clamp(0, 1) - x_j

    if was_training:
        model.train()

    adv_i = (x_i + delta_i).clamp(0, 1).detach()
    adv_j = (x_j + delta_j).clamp(0, 1).detach()
    return adv_i, adv_j


class AdversarialWeightPerturbation:
    """Eq. 9-10 (AWP, Wu et al. 2020, as used by TDLF). Perturbs weight
    tensors (ndim > 1 only -- skips biases/norm params for stability) in the
    direction that increases the current loss, scaled proportionally to
    each tensor's own norm.

    Usage per training step:
        optimizer.zero_grad()
        loss = criterion(...); loss.backward()   # grad at clean weights w
        awp.perturb()                            # w <- w + v
        optimizer.zero_grad()
        loss2 = criterion(...); loss2.backward() # grad at perturbed weights w+v
        awp.restore()                            # w+v <- w
        optimizer.step()                         # update w using grad from w+v
    """

    def __init__(self, model: nn.Module, gamma: float = 0.01, eps: float = 1e-6):
        self.model = model
        self.gamma = gamma
        self.eps = eps
        self._backup = {}

    def perturb(self):
        self._backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and param.ndim > 1:
                grad_norm = torch.norm(param.grad)
                if grad_norm == 0 or torch.isnan(grad_norm):
                    continue
                param_norm = torch.norm(param.data)
                perturbation = self.gamma * param_norm / (grad_norm + self.eps) * param.grad
                self._backup[name] = param.data.clone()
                param.data.add_(perturbation)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}
