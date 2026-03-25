"""Backtesting module for LOB prediction models.

Converts classification predictions (Down/Stationary/Up) into trading positions
and evaluates financial performance with realistic transaction costs.

📚 Key concepts explained:

**Signal-to-Trade Logic**
    A classifier says "price will go up/down/stay." A backtest asks: "If I traded
    on that signal, would I make money?" We convert predictions to positions:
    - Down (class 0) → Short (-1): bet that price falls, profit from decline
    - Stationary (class 1) → Flat (0): no position, no risk
    - Up (class 2) → Long (+1): bet that price rises, profit from increase

**PnL (Profit and Loss)**
    At each timestep, PnL = position × (next_price - current_price).
    If you're long (+1) and price goes up, you make money. If it goes down, you lose.
    If you're short (-1), it's reversed.

**Transaction Costs**
    Every time you change position, you pay the bid-ask spread. Buying costs more
    than the mid-price (you pay the ask), selling gets less (you receive the bid).
    Cost per unit of position change = spread / 2 (half-spread).
    Going flat→long costs spread/2. Going short→long costs spread (2 units of change).

**Sharpe Ratio**
    Sharpe = mean(returns) / std(returns). Measures risk-adjusted returns.
    A strategy with high returns but wild swings has a low Sharpe.
    Sharpe > 1 is considered good, > 2 is excellent. Here we compute per-step
    Sharpe since FI-2010 timesteps don't map cleanly to calendar time.

**Maximum Drawdown**
    The worst peak-to-trough decline in cumulative PnL. If your strategy made $100,
    then lost $30, then recovered — your max drawdown is 30%. Shows the worst pain
    an investor would endure.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Signal-to-trade conversion
# ---------------------------------------------------------------------------

def predictions_to_positions(predictions: np.ndarray) -> np.ndarray:
    """Convert classification predictions to trading positions.

    Args:
        predictions: Array of class labels {0=Down, 1=Stationary, 2=Up}.

    Returns:
        positions: Array of {-1=Short, 0=Flat, +1=Long}.
    """
    return predictions.astype(np.int64) - 1


# ---------------------------------------------------------------------------
# Mid-price and spread extraction
# ---------------------------------------------------------------------------

def extract_mid_prices_and_spreads(test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract mid-prices and bid-ask spreads from LOB test data.

    The FI-2010 features are INTERLEAVED by level (4 features per level):
        Index 0: ask_price_1, 1: ask_vol_1, 2: bid_price_1, 3: bid_vol_1
        Index 4: ask_price_2, 5: ask_vol_2, 6: bid_price_2, 7: bid_vol_2
        ... (10 levels total, 40 features)

    We use the LAST timestep of each window (index -1) since that's the
    "current" LOB state when the model makes its prediction.

    Note: Prices are DecPre-normalized (linearly scaled). Absolute PnL values
    are in normalized units, but Sharpe ratios and relative comparisons between
    models are valid since the scaling is identical for all.

    Args:
        test_x: Test features, shape (N, 100, 40).

    Returns:
        mid_prices: Shape (N,). Mid-price = (best_ask + best_bid) / 2.
        spreads: Shape (N,). Spread = best_ask - best_bid.
    """
    # Best ask price = feature 0, best bid price = feature 2 (interleaved layout)
    best_ask = test_x[:, -1, 0]  # last timestep, ask_price_level1
    best_bid = test_x[:, -1, 2]  # last timestep, bid_price_level1

    mid_prices = (best_ask + best_bid) / 2.0
    spreads = best_ask - best_bid

    return mid_prices, spreads


# ---------------------------------------------------------------------------
# PnL computation
# ---------------------------------------------------------------------------

def compute_pnl(
    positions: np.ndarray,
    mid_prices: np.ndarray,
    spreads: np.ndarray,
    cost_multiplier: float = 0.5,
) -> dict:
    """Compute PnL with transaction costs.

    At each timestep t (for t = 0, 1, ..., N-2):
        - We hold position[t]
        - Price moves from mid_price[t] to mid_price[t+1]
        - Gross PnL(t) = position[t] × (mid_price[t+1] - mid_price[t])
        - If position changed: cost = |pos[t] - pos[t-1]| × spread[t] × cost_multiplier
        - Net PnL(t) = Gross PnL(t) - cost(t)

    📚 Why cost_multiplier=0.5?
       When you buy at the ask and sell at the bid, the round-trip cost is the
       full spread. But each leg (buy OR sell) costs half the spread. So each
       unit of position change costs spread × 0.5.

    Args:
        positions: Trading positions, shape (N,). Values in {-1, 0, +1}.
        mid_prices: Mid-prices, shape (N,).
        spreads: Bid-ask spreads, shape (N,).
        cost_multiplier: Fraction of spread charged per unit position change.

    Returns:
        Dictionary with:
        - gross_pnl: Per-step gross PnL, shape (N-1,)
        - costs: Per-step transaction costs, shape (N-1,)
        - net_pnl: Per-step net PnL (gross - costs), shape (N-1,)
        - cumulative_gross: Cumulative gross PnL, shape (N-1,)
        - cumulative_net: Cumulative net PnL, shape (N-1,)
    """
    N = len(positions)
    assert len(mid_prices) == N and len(spreads) == N

    # Price changes: mid_price[t+1] - mid_price[t]
    price_changes = np.diff(mid_prices)  # shape (N-1,)

    # Gross PnL: position[t] × price_change[t]
    # We use positions[:-1] because we hold position[t] during [t, t+1)
    gross_pnl = positions[:-1] * price_changes

    # Position changes (for transaction costs)
    # At t=0, we go from flat (0) to positions[0]
    pos_changes = np.empty(N - 1)
    pos_changes[0] = abs(positions[0])
    pos_changes[1:] = np.abs(np.diff(positions[:-1]))

    # Transaction costs
    costs = pos_changes * spreads[:-1] * cost_multiplier

    # Net PnL
    net_pnl = gross_pnl - costs

    return {
        "gross_pnl": gross_pnl,
        "costs": costs,
        "net_pnl": net_pnl,
        "cumulative_gross": np.cumsum(gross_pnl),
        "cumulative_net": np.cumsum(net_pnl),
    }


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------

def compute_metrics(pnl_result: dict, positions: np.ndarray) -> dict:
    """Compute financial performance metrics.

    Args:
        pnl_result: Output from compute_pnl().
        positions: Trading positions, shape (N,).

    Returns:
        Dictionary of metrics.
    """
    net_pnl = pnl_result["net_pnl"]
    gross_pnl = pnl_result["gross_pnl"]
    cum_net = pnl_result["cumulative_net"]
    total_costs = pnl_result["costs"].sum()

    # --- Number of trades ---
    # A "trade" happens when position changes (including the initial position)
    pos_changes = np.diff(positions)
    num_trades = int(np.count_nonzero(pos_changes)) + (1 if positions[0] != 0 else 0)

    # --- Sharpe ratio (per-step) ---
    # mean / std of per-step net returns
    if net_pnl.std() > 0:
        sharpe = net_pnl.mean() / net_pnl.std()
    else:
        sharpe = 0.0

    # --- Maximum drawdown ---
    # Worst peak-to-trough decline in cumulative PnL
    running_max = np.maximum.accumulate(cum_net)
    drawdowns = running_max - cum_net
    max_drawdown = drawdowns.max()

    # --- Profit per trade ---
    profit_per_trade = cum_net[-1] / num_trades if num_trades > 0 else 0.0

    # --- Win rate ---
    # Fraction of steps with positive net PnL (excluding zero-PnL steps where flat)
    active_steps = net_pnl[positions[:-1] != 0]  # only steps where we had a position
    if len(active_steps) > 0:
        win_rate = (active_steps > 0).mean()
    else:
        win_rate = 0.0

    # --- Position statistics ---
    n = len(positions)
    frac_long = (positions == 1).sum() / n
    frac_short = (positions == -1).sum() / n
    frac_flat = (positions == 0).sum() / n

    return {
        "total_pnl_gross": gross_pnl.sum(),
        "total_pnl_net": cum_net[-1],
        "total_costs": total_costs,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "num_trades": num_trades,
        "profit_per_trade": profit_per_trade,
        "win_rate": win_rate,
        "frac_long": frac_long,
        "frac_short": frac_short,
        "frac_flat": frac_flat,
    }


# ---------------------------------------------------------------------------
# Convenience: run full backtest for a model
# ---------------------------------------------------------------------------

def run_backtest(
    predictions: np.ndarray,
    mid_prices: np.ndarray,
    spreads: np.ndarray,
    cost_multiplier: float = 0.5,
) -> dict:
    """Run a complete backtest for one model's predictions.

    Args:
        predictions: Model predictions, shape (N,). Values in {0, 1, 2}.
        mid_prices: Mid-prices, shape (N,).
        spreads: Bid-ask spreads, shape (N,).
        cost_multiplier: Transaction cost parameter.

    Returns:
        Dictionary with 'positions', 'pnl', and 'metrics' keys.
    """
    positions = predictions_to_positions(predictions)
    pnl = compute_pnl(positions, mid_prices, spreads, cost_multiplier)
    metrics = compute_metrics(pnl, positions)

    return {
        "positions": positions,
        "pnl": pnl,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Advanced strategies (probability-based)
# ---------------------------------------------------------------------------

def apply_confidence_filter(
    probabilities: np.ndarray,
    threshold: float = 0.7,
) -> np.ndarray:
    """Filter predictions by softmax confidence threshold.

    Only trade when max(softmax probability) > threshold. Low-confidence
    predictions are mapped to Stationary (class 1 → flat position).

    📚 Why does this help?
       Neural networks are often overconfident, but there's still a correlation
       between softmax probability and actual correctness. By only acting on
       high-confidence predictions, we filter out noisy signals and dramatically
       reduce trade count — which cuts transaction costs.

       The BDLOB paper (Zhang, Zohren & Roberts, 2018) found α=0.7 optimal
       on London Stock Exchange data, achieving 2-4× improvement in risk-adjusted
       returns versus trading on every signal.

    Args:
        probabilities: Softmax probability vectors, shape (N, 3).
            Columns are [P(Down), P(Stationary), P(Up)].
        threshold: Minimum max-probability to generate a trade signal.
            Predictions below this threshold become Stationary.

    Returns:
        predictions: Filtered class labels {0, 1, 2}, shape (N,).
    """
    max_probs = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    predictions[max_probs <= threshold] = 1  # Map to Stationary
    return predictions


def apply_holding_period(
    predictions: np.ndarray,
    min_hold: int = 10,
) -> np.ndarray:
    """Enforce minimum holding period on trading signals.

    Once a directional signal (Up/Down) is generated, hold that position
    for at least min_hold timesteps, ignoring intermediate signals.

    📚 Why does this help?
       A model trained at horizon k=10 predicts price movement over the next
       10 timesteps. Re-evaluating at every tick treats overlapping windows
       as independent — but prediction at t (covering [t, t+10]) and at t+1
       (covering [t+1, t+11]) share 90% of their information. Holding for k
       steps aligns trading frequency with the prediction horizon, reducing
       unnecessary position changes and their associated costs.

    Args:
        predictions: Class labels {0=Down, 1=Stationary, 2=Up}, shape (N,).
        min_hold: Minimum timesteps to hold a directional position.

    Returns:
        filtered: Modified predictions with holding period enforced.
    """
    result = predictions.copy()
    hold_remaining = 0
    current_signal = 1  # Start flat

    for i in range(len(result)):
        if hold_remaining > 0:
            result[i] = current_signal
            hold_remaining -= 1
        elif result[i] != 1:  # New directional signal
            current_signal = result[i]
            hold_remaining = min_hold - 1
        # If Stationary and not holding → stay flat (no change needed)

    return result


def ensemble_predictions(
    prob_list: list[np.ndarray],
    min_agreement: int | None = None,
    confidence_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine multiple models' probabilities into ensemble prediction.

    Averages softmax probabilities across models, then takes argmax.
    Optionally requires minimum model agreement (consensus filter).

    📚 How ensemble consensus works:
       When 4/4 models independently predict "Up," the signal is much more
       reliable than when 2 predict "Up" and 2 predict "Stationary." By
       requiring consensus, we filter to only the highest-conviction trades.

       This is a different use case than improving accuracy — it's about
       filtering for trade quality. Even if ensemble accuracy ≈ best single
       model, the consensus filter dramatically reduces low-quality trades.

    Args:
        prob_list: List of M probability arrays, each shape (N, 3).
        min_agreement: Require at least this many models to agree on the
            predicted class. Disagreements → Stationary.
        confidence_threshold: Optional threshold on the averaged probability
            (applied after averaging). None = no threshold.

    Returns:
        predictions: Ensemble class labels {0, 1, 2}, shape (N,).
        avg_probs: Averaged probability vectors, shape (N, 3).
    """
    stacked = np.stack(prob_list)        # (M, N, 3)
    avg_probs = stacked.mean(axis=0)     # (N, 3)
    predictions = avg_probs.argmax(axis=1).astype(np.int64)

    if min_agreement is not None:
        # Count how many models agree with the ensemble prediction
        individual_preds = stacked.argmax(axis=2)  # (M, N)
        agreement = (individual_preds == predictions[np.newaxis, :]).sum(axis=0)
        predictions[agreement < min_agreement] = 1  # Stationary

    if confidence_threshold is not None:
        max_probs = avg_probs.max(axis=1)
        predictions[max_probs <= confidence_threshold] = 1

    return predictions, avg_probs
