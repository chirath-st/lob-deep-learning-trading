"""Cost-aware loss functions for LOB prediction models.

Standard cross-entropy treats all misclassifications equally. But in trading:
- Predicting "up" when market goes "down" = realized loss (buy, price drops)
- Predicting "up" when market is flat = unnecessary trade (pay the spread)
- Predicting "flat" when market goes "up" = missed opportunity (no cost)

CE gives equal weight to all three errors, but only the first two COST money.
The loss functions here fix this asymmetry in different ways.

This module provides three alternative loss functions:

1. FocalLoss
   Down-weights easy/confident predictions. Forces the model to focus on
   hard boundary cases where UP/DOWN meets STATIONARY.

2. TurnoverPenalizedLoss
   CE + penalty for changing predictions between consecutive timesteps.
   Directly attacks the excessive-trading problem by encouraging smooth
   prediction sequences.

3. DifferentiableSharpeLoss
   Bypasses classification entirely. Converts model outputs to continuous
   positions, simulates net-of-cost returns, and optimizes the Sharpe ratio
   directly via backpropagation.

References:
    - Focal Loss: Lin et al. (2017), "Focal Loss for Dense Object Detection"
    - Turnover reg: Lim, Zohren & Roberts (2019), "Deep Momentum Networks"
    - Diff. Sharpe: Zhang, Zohren & Roberts (2020), JFDS
    - Half-cost trick: Wood, Roberts & Zohren (2025), arXiv:2601.05975
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss — down-weights easy examples to focus on hard ones.

    Standard cross-entropy for a correctly-classified sample with probability
    p_t is: L_CE = -log(p_t).

    Focal loss adds a modulating factor:

        L_FL = -(1 - p_t)^gamma * log(p_t)

    How the modulation works (with gamma=2):
        p_t = 0.9 (easy, correct):  (1-0.9)^2 = 0.01  -> loss reduced 100x
        p_t = 0.5 (uncertain):      (1-0.5)^2 = 0.25  -> loss reduced 4x
        p_t = 0.1 (hard, wrong):    (1-0.1)^2 = 0.81  -> loss barely reduced

    The gamma parameter controls how aggressive the down-weighting is:
        gamma = 0: Standard cross-entropy (no focal effect)
        gamma = 1: Moderate focusing
        gamma = 2: Strong focusing (recommended starting point)
        gamma = 5: Very aggressive — only hardest examples matter

    Why this helps LOB trading:
        The stationary class dominates FI-2010 (~50-60% at short horizons).
        Standard CE spends most gradient signal getting these easy predictions
        right. Focal loss shifts attention to the boundary cases between
        UP/DOWN and STATIONARY — exactly where trading decisions are made.

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
        alpha: Optional per-class weights, shape (num_classes,).
        reduction: 'mean', 'sum', or 'none'.

    Reference:
        Lin et al. (2017), "Focal Loss for Dense Object Detection", ICCV
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model output, shape (B, C). NOT softmax probabilities.
            targets: Class labels, shape (B,). Integer values in [0, C).

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or per-sample loss (B,).
        """
        # log_softmax is numerically more stable than log(softmax(x))
        log_probs = F.log_softmax(logits, dim=1)  # (B, C)
        probs = torch.exp(log_probs)  # (B, C)

        # Get probability and log-probability of the TRUE class for each sample.
        # gather(dim=1, index) selects one value per row using the index.
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)

        # Focal modulation: (1 - p_t)^gamma
        # Confident correct predictions (p_t -> 1) get factor -> 0
        # Uncertain predictions (p_t -> 0) get factor -> 1
        focal_weight = (1 - p_t) ** self.gamma  # (B,)

        # Optional per-class weights
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # (B,)
            focal_weight = alpha_t * focal_weight

        # Final loss: -weight * log(p_t)
        loss = -focal_weight * log_p_t  # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TurnoverPenalizedLoss(nn.Module):
    """Cross-Entropy + Turnover Penalty — discourages rapid prediction changes.

    Total loss:  L = L_CE + lambda * L_turnover

    The turnover penalty works by:

    1. Converting logits to "soft positions" via softmax:
       pos_t = P(Up)_t - P(Down)_t    (in [-1, 1])

       This maps the 3-class probabilities to a continuous signal:
       pos ~ +1: strong buy   (high P(Up))
       pos ~  0: flat         (high P(Stationary) or uncertain)
       pos ~ -1: strong sell  (high P(Down))

    2. Computing turnover between consecutive timesteps:
       turnover_t = |pos_t - pos_{t-1}|

    3. Penalizing the average turnover across the batch:
       L_turnover = mean(turnover)

    Why this works:
        Adjacent LOB snapshots (t and t+1) share ~99% of their information.
        The model's predictions SHOULD be smooth over time. But standard CE
        treats each sample independently with no incentive for smoothness.
        The turnover penalty adds that incentive: "if you change your mind,
        that change better be worth the penalty."

    The lambda parameter controls the smoothness-accuracy tradeoff:
        lambda = 0:    Pure CE (no smoothing)
        lambda = 0.01: Gentle smoothing
        lambda = 0.1:  Moderate smoothing (good starting point)
        lambda = 1.0:  Heavy smoothing (may hurt accuracy)

    IMPORTANT: Requires sequential (non-shuffled) data loading!
    Consecutive samples in the batch must be consecutive in time.

    Args:
        lambda_turnover: Weight for the turnover penalty term.

    Reference:
        Lim, Zohren & Roberts (2019), "Deep Momentum Networks", JFDS
    """

    def __init__(self, lambda_turnover: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_turnover = lambda_turnover

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute CE + turnover penalty.

        Args:
            logits: Raw model output, shape (B, 3). Samples must be in
                    chronological order within the batch.
            targets: Class labels, shape (B,).

        Returns:
            total_loss: Scalar loss for backpropagation.
            components: Dict with 'ce_loss' and 'turnover' for logging.
        """
        ce_loss = self.ce(logits, targets)

        # Convert to soft positions: P(Up) - P(Down)
        probs = F.softmax(logits, dim=1)  # (B, 3)
        positions = probs[:, 2] - probs[:, 0]  # (B,)

        # Turnover between consecutive samples
        if len(positions) > 1:
            pos_diff = positions[1:] - positions[:-1]
            # Smooth absolute value: sqrt(x^2 + eps) instead of |x|
            # This avoids the non-differentiable point at x=0
            turnover = torch.sqrt(pos_diff**2 + 1e-8).mean()
        else:
            turnover = torch.tensor(0.0, device=logits.device, requires_grad=True)

        total_loss = ce_loss + self.lambda_turnover * turnover

        return total_loss, {
            "ce_loss": ce_loss.item(),
            "turnover": turnover.item(),
        }


class DifferentiableSharpeLoss(nn.Module):
    """Differentiable Sharpe Ratio Loss — optimizes risk-adjusted returns directly.

    Instead of: Classify -> Decide trades -> Evaluate profits
    We make the entire pipeline differentiable:
        Model output -> Soft positions -> Simulated returns -> Sharpe -> Backprop

    Step by step:

    1. POSITIONS from model output:
       pos_t = P(Up)_t - P(Down)_t    (in [-1, 1])
       Continuous relaxation of discrete {-1, 0, +1} trading positions.

    2. RETURNS from positions and price changes:
       R_t = pos_t * (price_{t+1} - price_t)
       Long + price up = profit. Short + price down = profit.

    3. TRANSACTION COSTS from position changes:
       TC_t = gamma_cost * spread_t * |pos_t - pos_{t-1}|

       gamma_cost is the "half-cost trick" parameter (Wood et al., 2025):
       - gamma = 0:   No cost awareness -> excessive trades
       - gamma = 1.0: Full cost -> model learns to NEVER trade (inertia trap)
       - gamma = 0.5: Sweet spot — cost-aware but not paralyzed
       This single trick improved net Sharpe by ~50% over full-cost training.

    4. NET RETURNS:
       NR_t = R_t - TC_t

    5. SHARPE RATIO (maximize this):
       Sharpe = mean(NR) / std(NR)
       Why Sharpe, not just total return? Maximizing total return encourages
       huge risky bets. Sharpe penalizes variance — it wants CONSISTENT
       returns, which is what real traders optimize for.

    6. LOSS:
       L = -Sharpe + lambda * mean(|delta_pos|)
       Negate Sharpe because PyTorch minimizes loss but we want to maximize.
       The turnover penalty provides additional smoothing beyond the cost term.

    IMPORTANT:
        - Requires sequential (non-shuffled) data loading
        - Best results with warm-starting from a pre-trained CE model
        - Use a lower learning rate (e.g., 0.0001) to preserve learned features

    Args:
        cost_multiplier: Fraction of spread per position change (0.5 = half-spread).
        gamma_cost: Cost scaling for training. 0.5 recommended (half-cost trick).
        lambda_turnover: Additional turnover regularization weight.
        eps: Small constant for numerical stability.

    References:
        Moody & Saffell (2001), IEEE TNN (original differentiable Sharpe)
        Zhang, Zohren & Roberts (2020), JFDS
        Wood, Roberts & Zohren (2025), arXiv:2601.05975 (half-cost trick)
    """

    def __init__(
        self,
        cost_multiplier: float = 0.5,
        gamma_cost: float = 0.5,
        lambda_turnover: float = 0.01,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.cost_multiplier = cost_multiplier
        self.gamma_cost = gamma_cost
        self.lambda_turnover = lambda_turnover
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        mid_prices: torch.Tensor,
        spreads: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute differentiable Sharpe ratio loss.

        Args:
            logits: Raw model output, shape (B, 3). Sequential order required.
            mid_prices: Mid-prices for each sample, shape (B,).
            spreads: Bid-ask spreads for each sample, shape (B,).

        Returns:
            loss: Scalar loss for backpropagation.
            components: Dict with diagnostic values for logging.
        """
        B = logits.size(0)
        if B < 3:
            # Not enough samples for meaningful Sharpe computation
            return torch.tensor(0.0, device=logits.device, requires_grad=True), {
                "sharpe": 0.0,
                "avg_return": 0.0,
                "avg_cost": 0.0,
                "turnover": 0.0,
            }

        # Step 1: Soft positions from model output
        probs = F.softmax(logits, dim=1)  # (B, 3)
        positions = probs[:, 2] - probs[:, 0]  # (B,) in [-1, 1]

        # Step 2: Price changes between consecutive samples
        price_changes = mid_prices[1:] - mid_prices[:-1]  # (B-1,)

        # Step 3: Gross returns = position * price change
        gross_returns = positions[:-1] * price_changes  # (B-1,)

        # Step 4: Transaction costs
        # Smooth |x| = sqrt(x^2 + eps) for differentiability
        pos_changes = positions[1:] - positions[:-1]  # (B-1,)
        smooth_abs_changes = torch.sqrt(pos_changes**2 + self.eps)  # (B-1,)
        costs = (
            smooth_abs_changes
            * spreads[:-1]
            * self.cost_multiplier
            * self.gamma_cost
        )

        # Step 5: Net returns
        net_returns = gross_returns - costs  # (B-1,)

        # Step 6: Negative Sharpe ratio (we minimize, so negate the Sharpe we want to maximize)
        mean_return = net_returns.mean()
        std_return = net_returns.std()
        sharpe = mean_return / (std_return + self.eps)

        # Step 7: Turnover regularization (additional smoothing)
        turnover = smooth_abs_changes.mean()

        loss = -sharpe + self.lambda_turnover * turnover

        return loss, {
            "sharpe": sharpe.item(),
            "avg_return": mean_return.item(),
            "avg_cost": costs.mean().item(),
            "turnover": turnover.item(),
        }
