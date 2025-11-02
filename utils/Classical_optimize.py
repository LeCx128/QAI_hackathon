import numpy as np
import pandas as pd


# ---------------------------- 1. Baseline QP ----------------------------

import cvxpy as cp
from typing import Union # For type hinting

def solve_scalarized_qp(
    Sigma: Union[np.ndarray, pd.DataFrame],
    mu: Union[np.ndarray, pd.Series],
    esg: Union[np.ndarray, pd.Series],
    lamb: float = 1.0,
    eta: float = 0.1,
    lb: float = 0.0,
    ub: float = 1.0,
    allow_short: bool = False,
    allow_cash: bool = False, # New parameter
    solver = cp.OSQP
) -> np.ndarray:
    """Solves a convex quadratic portfolio optimization problem.

    This function finds the optimal portfolio weights 'w' that minimize a
    scalarized objective function combining portfolio variance (risk),
    expected return, and ESG score, subject to standard portfolio constraints.

    The objective function minimized is:
        w^T * Sigma * w - lamb * (mu^T * w) - eta * (esg^T * w)
    Effectively, it maximizes a utility function U = lamb*Return + eta*ESG - Risk.

    Args:
        Sigma (np.array | pd.DataFrame): Covariance matrix (n x n). Must be positive semi-definite.
        mu (np.array | pd.Series): Expected returns vector (n,).
        esg (np.array | pd.Series): ESG scores vector (n,).
        lamb (float, optional): Weight for the expected return term. Defaults to 1.0.
        eta (float, optional): Weight for the ESG score term. Defaults to 0.1.
        lb (float, optional): Lower bound for individual asset weights. Defaults to 0.0.
        ub (float, optional): Upper bound for individual asset weights. Defaults to 1.0.
        allow_short (bool, optional): If True, ignores the lb >= 0 constraint. Defaults to False.
        allow_cash (bool, optional): If True, constraints sum(weights) <= 1 instead of == 1,
                                     allowing for a portion of the portfolio to be uninvested (cash).
                                     Defaults to False (fully invested).
        solver (cp.Solver, optional): The CVXPY solver. Defaults to cp.OSQP.

    Returns:
        np.array: Optimal portfolio weights (shape n,).

    Raises:
        RuntimeError: If the solver fails.
        ValueError: If input dimensions are inconsistent.
    """
    # --- Input Validation and Conversion ---
    if isinstance(mu, (pd.Series, pd.DataFrame)):
        mu_np = mu.values.flatten()
    else:
        mu_np = np.array(mu).flatten()

    n = len(mu_np)

    if isinstance(esg, (pd.Series, pd.DataFrame)):
        esg_np = esg.values.flatten()
    else:
        esg_np = np.array(esg).flatten()

    if isinstance(Sigma, pd.DataFrame):
        Sigma_np = Sigma.values
    else:
        Sigma_np = np.array(Sigma)

    if len(esg_np) != n or Sigma_np.shape != (n, n):
         raise ValueError(
             f"Dimension mismatch: n={n}, mu={len(mu_np)}, esg={len(esg_np)}, Sigma={Sigma_np.shape}"
         )
    # ----------------------------------------

    w = cp.Variable(n, name="weights")

    # --- Objective Function ---
    objective = cp.quad_form(w, Sigma_np) - lamb * (mu_np @ w) - eta * (esg_np @ w)

    # --- Constraints ---
    constraints = []

    # Budget Constraint: sum(w) == 1 (fully invested) or sum(w) <= 1 (allow cash)
    if allow_cash:
        constraints.append(cp.sum(w) <= 1)
        # If allowing cash, weights must still be non-negative unless shorting is also allowed
        if not allow_short:
             constraints.append(w >= 0) # Ensure weights don't go negative to fund cash
    else:
        constraints.append(cp.sum(w) == 1)

    # Individual Weight Bounds
    if not allow_short:
        # If sum(w) <= 1 is used, lb might technically not be needed if >= 0 already added
        # but including it handles cases where lb > 0 is desired.
        constraints.append(w >= lb)
    # If allow_short is True, we don't add w >= lb constraint.
    # The upper bound always applies.
    constraints.append(w <= ub)

    # --- Solve Problem ---
    prob = cp.Problem(cp.Minimize(objective), constraints)
    try:
        prob.solve(solver=solver, verbose=False)
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
             raise RuntimeError(f'QP solver failed or did not find optimal solution. Status: {prob.status}')
        if w.value is None:
             raise RuntimeError('QP solver finished but weights are None.')

        result_w = np.array(w.value).flatten()
        # Clean near-zero values
        result_w[np.abs(result_w) < 1e-8] = 0.0
        return result_w
    except Exception as e:
        raise RuntimeError(f'QP solver failed during execution: {e}')



# ---------------------------- 2. CVaR optimization ----------------------------
import cvxpy as cp
from typing import Union # For type hinting

def solve_cvar_lp(
    returns_scenarios: Union[np.ndarray, pd.DataFrame],
    mu: Union[np.ndarray, pd.Series],
    esg: Union[np.ndarray, pd.Series],
    alpha: float = 0.95,
    lamb: float = 1.0,
    eta: float = 0.1,
    lb: float = 0.0,
    ub: float = 1.0,
    allow_short: bool = False,
    allow_cash: bool = False, # New parameter
    solver = cp.OSQP
) -> np.ndarray:
    """Solves portfolio optimization minimizing CVaR - lambda*Return - eta*ESG.

    This function uses a linear programming formulation (based on Rockafellar-Uryasev)
    to find the optimal portfolio weights 'w' that minimize a weighted combination
    of Conditional Value-at-Risk (CVaR) and negative expected return and ESG score.

    The objective function minimized is approximately:
        CVaR_{alpha}(Loss) - lamb * (mu^T * w) - eta * (esg^T * w)
    where Loss = -Return.

    Args:
        returns_scenarios (np.array | pd.DataFrame): Shape (T, n) of T scenarios
                                                    for n assets. Each row is one
                                                    scenario's returns.
        mu (np.array | pd.Series): Shape (n,) of expected returns.
        esg (np.array | pd.Series): Shape (n,) of ESG scores.
        alpha (float, optional): Confidence level for CVaR (e.g., 0.95 for 95% CVaR).
                                 Defaults to 0.95.
        lamb (float, optional): Weight for the expected return term in the objective.
                                Higher values prioritize return. Defaults to 1.0.
        eta (float, optional): Weight for the ESG term in the objective. Higher values
                               prioritize ESG. Defaults to 0.1.
        lb (float, optional): Lower bound for individual asset weights. Defaults to 0.0.
        ub (float, optional): Upper bound for individual asset weights. Defaults to 1.0.
        allow_short (bool, optional): If True, ignores the non-negativity implied by lb=0.0
                                      and allows weights to be negative. Defaults to False.
        allow_cash (bool, optional): If True, constraints sum(weights) <= 1 instead of == 1,
                                     allowing for a portion of the portfolio to be uninvested (cash).
                                     Defaults to False (fully invested).
        solver (cp.Solver, optional): CVXPY solver to use (e.g., cp.OSQP, cp.ECOS, cp.SCS).
                                      Defaults to cp.OSQP.

    Returns:
        np.array: Optimal portfolio weights, shape (n,).

    Raises:
        RuntimeError: If the solver fails or does not find an optimal solution.
        ValueError: If input dimensions are inconsistent.
    """
    # --- Input Validation and Conversion ---
    if isinstance(returns_scenarios, pd.DataFrame):
        returns_scenarios_np = returns_scenarios.values
    else:
        returns_scenarios_np = np.array(returns_scenarios)

    T, n = returns_scenarios_np.shape

    if isinstance(mu, (pd.Series, pd.DataFrame)):
        mu_np = mu.values.flatten()
    else:
        mu_np = np.array(mu).flatten()

    if isinstance(esg, (pd.Series, pd.DataFrame)):
        esg_np = esg.values.flatten()
    else:
        esg_np = np.array(esg).flatten()

    # Check dimensions
    if len(mu_np) != n or len(esg_np) != n:
        raise ValueError(
            f"Dimension mismatch: n={n} from scenarios, mu={len(mu_np)}, esg={len(esg_np)}"
        )
    # ----------------------------------------

    # --- CVXPY Variables ---
    w = cp.Variable(n, name="weights")
    v = cp.Variable(1, name="VaR_alpha")          # Auxiliary variable for VaR
    xi = cp.Variable(T, nonneg=True, name="losses_over_VaR") # Auxiliary variables for losses exceeding VaR

    # --- Calculate Portfolio Losses per Scenario ---
    portfolio_losses = -returns_scenarios_np @ w  # Shape (T,)

    # --- Core CVaR Constraint ---
    # xi_t >= Loss_t - v  (for all t=1..T)
    # Ensures xi captures the positive part of (Loss_t - v)
    cvar_core_constraint = xi >= portfolio_losses - v

    # --- Define Constraints List ---
    constraints = [
        w <= ub,                 # Upper bound on weights
        cvar_core_constraint     # The CVaR formulation constraint
        # xi >= 0 is handled by cp.Variable(T, nonneg=True)
    ]

    # --- Budget Constraint (Allow Cash or Fully Invested) ---
    if allow_cash:
        constraints.append(cp.sum(w) <= 1)
        # If allowing cash, weights usually must still be non-negative
        # unless shorting is *also* allowed. Add non-negativity if needed.
        if not allow_short:
             constraints.append(w >= 0) # Prevent negative weights funding cash
    else:
        # Fully invested
        constraints.append(cp.sum(w) == 1)

    # --- Lower Bound / Short Selling Constraint ---
    if not allow_short:
        # Add the lower bound constraint (potentially redundant if allow_cash added w >= 0)
        # but handles cases where lb > 0 is desired.
        constraints.append(w >= lb)
    # If allow_short is True, we simply don't add w >= lb.

    # --- Objective Function ---
    # CVaR = VaR + Average Tail Loss
    cvar_term = v + (1.0 / ((1 - alpha) * T)) * cp.sum(xi)
    # Minimize: CVaR - lambda * Expected Return - eta * ESG Score
    objective = cvar_term - lamb * (mu_np @ w) - eta * (esg_np @ w)

    # --- Problem Setup & Solving ---
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    if w.value is None:
        raise RuntimeError('CVaR LP solver failed')
    return np.array(w.value).reshape(-1,)


# ---------------------------- Evaluation metrics & plotting ----------------------------
from typing import Optional, Dict, Union, Tuple, Any

def calculate_portfolio_metrics(
    weights: np.ndarray,
    expected_returns: Union[np.ndarray, pd.Series],
    covariance_matrix: Union[np.ndarray, pd.DataFrame],
    esg_scores: Union[np.ndarray, pd.Series],
    returns_oos: Optional[Union[pd.DataFrame, pd.Series]] = None,
    annualization_factor: int = 252
    ) -> Dict[str, float]:
    """
    Calculates various performance and characteristic metrics for a given portfolio.

    Computes the expected return, risk (variance and volatility), and ESG score
    based on input parameters. Optionally calculates realized out-of-sample (OOS)
    mean return and annualized volatility if OOS returns are provided.

    Args:
        weights (np.ndarray): A 1D array of portfolio weights for n assets (shape n,).
            Weights should ideally sum to 1.
        expected_returns (Union[np.ndarray, pd.Series]): A 1D array or Series of
            expected returns for each asset (shape n,).
        covariance_matrix (Union[np.ndarray, pd.DataFrame]): The covariance matrix
            of asset returns (shape n x n).
        esg_scores (Union[np.ndarray, pd.Series]): A 1D array or Series of ESG
            scores for each asset (shape n,).
        returns_oos (Optional[Union[pd.DataFrame, pd.Series]], optional):
            Out-of-sample historical or simulated returns used to calculate realized
            performance. Should be a DataFrame (T x n) or Series if only one asset,
            where T is the number of periods. Assumes daily returns if using the
            default annualization_factor. Defaults to None.
        annualization_factor (int, optional): The factor used to annualize the
            out-of-sample volatility (e.g., 252 for daily returns, 12 for monthly).
            Defaults to 252.

    Returns:
        Dict[str, float]: A dictionary containing the calculated portfolio metrics:
            - 'expected_return': The expected portfolio return (annualized if inputs are).
            - 'variance': The portfolio variance.
            - 'volatility': The portfolio standard deviation (square root of variance).
            - 'esg_score': The weighted average ESG score of the portfolio.
            - 'oos_mean_return' (optional): The mean of the out-of-sample portfolio returns.
            - 'oos_annualized_volatility' (optional): The annualized standard deviation
              (volatility) of the out-of-sample portfolio returns.

    Raises:
        ValueError: If input dimensions are inconsistent.
    """
    # --- Input Validation and Conversion ---
    n_assets = len(weights)

    if isinstance(expected_returns, pd.Series):
        mu = expected_returns.values
    else:
        mu = np.asarray(expected_returns)

    if isinstance(covariance_matrix, pd.DataFrame):
        Sigma = covariance_matrix.values
    else:
        Sigma = np.asarray(covariance_matrix)

    if isinstance(esg_scores, pd.Series):
        esg = esg_scores.values
    else:
        esg = np.asarray(esg_scores)

    if mu.shape != (n_assets,) or esg.shape != (n_assets,) or Sigma.shape != (n_assets, n_assets):
        raise ValueError(
            f"Input dimension mismatch: weights({weights.shape}), mu({mu.shape}), "
            f"esg({esg.shape}), Sigma({Sigma.shape})"
        )
    if not np.isclose(np.sum(weights), 1.0):
        print("Warning: Portfolio weights do not sum close to 1.")

    # --- Calculate Core Metrics ---
    port_expected_return = float(mu @ weights)
    port_variance = float(weights.T @ Sigma @ weights)
    # Ensure variance is non-negative due to potential floating point errors
    port_variance = max(0, port_variance)
    port_volatility = np.sqrt(port_variance)
    port_esg_score = float(esg @ weights)

    metrics = {
        'expected_return': port_expected_return,
        'variance': port_variance,
        'volatility': port_volatility,
        'esg_score': port_esg_score
    }

    # --- Calculate Out-of-Sample Metrics (Optional) ---
    if returns_oos is not None:
        if isinstance(returns_oos, (pd.DataFrame, pd.Series)):
            oos_returns_np = returns_oos.values
        else:
            oos_returns_np = np.asarray(returns_oos)

        # Handle potential 1D array for single asset OOS returns
        if oos_returns_np.ndim == 1:
             if n_assets != 1:
                 raise ValueError("returns_oos is 1D but multiple assets exist.")
             # Reshape for consistent calculation
             oos_returns_np = oos_returns_np.reshape(-1, 1)
        elif oos_returns_np.shape[1] != n_assets:
             raise ValueError(
                 f"Dimension mismatch for returns_oos: Expected {n_assets} columns, "
                 f"got {oos_returns_np.shape[1]}"
             )

        # Calculate portfolio returns for each OOS period
        portfolio_oos_returns = oos_returns_np @ weights

        # Calculate OOS metrics
        oos_mean = float(np.mean(portfolio_oos_returns))
        oos_vol = float(np.std(portfolio_oos_returns))
        oos_annualized_vol = oos_vol * np.sqrt(annualization_factor)

        metrics['oos_mean_return'] = oos_mean
        metrics['oos_annualized_volatility'] = oos_annualized_vol

    return metrics

# ---------------------------- Portfolio Optimization Suite ----------------------------

'''
- Remove prices
- Change esg_s to esg_s

- Add asset_arr
- Add mu
- Add Sigma
'''

def run_portfolio_optimization_suite(
    price_df,
    mu,
    Sigma,
    asset_arr,
    esg_s  : pd.Series,
    config: Dict[str, Any],
    oos_prices = None
) -> Dict[str, Any]:
    """
    Runs a suite of portfolio optimization methods based on the provided configuration.

    This function takes price and ESG data, computes necessary inputs (expected
    returns, covariance matrix), and then executes various optimization
    strategies as specified in the configuration dictionary. It evaluates
    the resulting portfolios and returns a structured dictionary of results.

    Args:
        price_df (pd.DataFrame): DataFrame of historical asset price_df (rows=dates, cols=asset_arr).
        esg_s (pd.Series): DataFrame of ESG scores (rows=asset_arr, cols=score).
                                  Must have an index compatible with price columns.
        config (Dict[str, Any]): A dictionary controlling which optimizations to run
                                 and their parameters. Expected keys:
            'compute_inputs': { 'shrinkage': bool }
            'scalarized_qp' (optional): {
                'lamb_grid': List[float],
                'eta_grid': List[float],
                'lb': float, 'ub': float, 'allow_short': bool, 'solver': cp.Solver
            }
            'nsga2' (optional): {
                'pop_size': int, 'generations': int, 'plot': bool
            }
            'cardinality' (optional): {
                'k': int, 'lamb': float, 'eta': float
            }
            'cvar' (optional): {
                'alpha': float, 'lamb': float, 'eta': float, 'use_mu_in_obj': bool
            }
        oos_prices (Optional[pd.DataFrame], optional): Out-of-sample price_df for
                                                      evaluating realized metrics.
                                                      Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing the results of the executed
                        optimization strategies. Keys correspond to strategy names
                        (e.g., 'scalarized_qp_results', 'nsga2_result', etc.).
    """
    results = {}
    # Align ESG scores with the asset_arr used (e.g., after dropping NaNs)
    try:
        esg_arr = esg_s.loc[asset_arr].values.flatten()
    except KeyError:
        raise ValueError("ESG scores index does not match price columns.")
    if len(esg_arr) != len(asset_arr):
         raise ValueError("ESG scores could not be aligned with asset_arr.")

    results['inputs'] = {'mu': mu, 'Sigma': Sigma, 'esg_arr': esg_arr, 'asset_arr': asset_arr}
    print(f"Inputs computed for {len(asset_arr)} asset_arr.")

    # --- Out-of-Sample Returns (if applicable) ---
    returns_oos_df = None
    if oos_prices is not None:
        returns_oos_df = oos_prices.pct_change().dropna()
        # Ensure OOS returns align with the asset_arr used in optimization
        try:
            returns_oos_df = returns_oos_df[asset_arr]
        except KeyError:
             print("Warning: OOS price columns do not perfectly match optimized asset_arr. Using intersection.")
             common_assets = asset_arr.intersection(returns_oos_df.columns)
             returns_oos_df = returns_oos_df[common_assets]
             # Re-align other inputs - this is complex, better to ensure data matches upfront
             # For simplicity here, we'll proceed but metrics might be slightly off
             # A robust implementation might filter asset_arr earlier or raise an error

    # --- 1. Scalarized QP Grid Search ---
    if 'scalarized_qp' in config:
        print("\nRunning Scalarized QP grid search...")
        qp_config = config['scalarized_qp']
        qp_results = []
        for lamb in qp_config.get('lamb_grid', [1.0]):
            for eta in qp_config.get('eta_grid', [0.1]):
                try:
                    w = solve_scalarized_qp(
                        Sigma, mu, esg_arr,
                        lamb=lamb, eta=eta,
                        lb=qp_config.get('lb', 0.0),
                        ub=qp_config.get('ub', 1.0),
                        allow_short=qp_config.get('allow_short', False),
                        allow_cash=qp_config.get('allow_cash', False),
                        solver=qp_config.get('solver', cp.OSQP)
                    )
                    metrics = calculate_portfolio_metrics(w, mu, Sigma, esg_arr, returns_oos=returns_oos_df)
                    qp_results.append({'lambda': lamb, 'eta': eta, 'weights': w, **metrics})
                    print(f"  lambda={lamb:.2f}, eta={eta:.2f} -> "
                          f"ret={metrics['expected_return']:.4f}, "
                          f"vol={metrics['volatility']:.4f}, "
                          f"esg={metrics['esg_score']:.4f}")
                except Exception as e:
                    print(f"  Failed for lambda={lamb}, eta={eta}: {e}")
        results['scalarized_qp_results'] = qp_results

    # --- 2. CVaR Optimization ---
    if 'cvar' in config:
        print("\nRunning CVaR Optimization...")
        cvar_config = config['cvar']
        daily_returns = price_df.dropna().pct_change().dropna()
         # Align scenarios with asset_arr used in mu/Sigma calculation
        try:
            scenarios = daily_returns[asset_arr].values
            if scenarios.shape[1] != len(asset_arr):
                 raise ValueError("Scenario dimensions don't match asset_arr")

            # Decide whether to include mu term based on config
            cvar_lamb = cvar_config.get('lamb', 0.0) if cvar_config.get('use_mu_in_obj', False) else 0.0

            w_cvar = solve_cvar_lp(
                scenarios, mu, esg_arr, # Pass mu here
                alpha=cvar_config.get('alpha', 0.95),
                lamb=cvar_lamb, # Pass lambda for return term
                eta=cvar_config.get('eta', 0.1),
                lb=cvar_config.get('lb', 0.0),
                ub=cvar_config.get('ub', 1.0),
                allow_short=cvar_config.get('allow_short', False),
                allow_cash=qp_config.get('allow_cash', False),
                solver=cvar_config.get('solver', cp.OSQP)
            )
            metrics_cvar = calculate_portfolio_metrics(w_cvar, mu, Sigma, esg_arr, returns_oos=returns_oos_df)
            print(f"  CVaR alpha={cvar_config.get('alpha', 0.95)}, "
                  f"return lambda={cvar_lamb:.2f}, "
                  f"ESG eta={cvar_config.get('eta', 0.1):.2f}")
            print(f"  Non-zero weights: {np.sum(np.abs(w_cvar) > 1e-6)}")
            print(f"  CVaR Metrics: ret={metrics_cvar['expected_return']:.4f}, "
                  f"vol={metrics_cvar['volatility']:.4f}, "
                  f"esg={metrics_cvar['esg_score']:.4f}")
            results['cvar_result'] = {'weights': w_cvar, 'metrics': metrics_cvar}

        except Exception as e:
            print(f"  CVaR optimization failed: {e}")

    print("\nOptimization suite finished.")
    return results