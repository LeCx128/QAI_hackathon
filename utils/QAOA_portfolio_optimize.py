from typing import List, Tuple, Union, Optional

import numpy as np
from docplex.mp.advmodel import AdvModel

from qiskit.result import QuasiDistribution
from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_finance.exceptions import QiskitFinanceError

from qiskit_aer.primitives import Sampler
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import matplotlib.pyplot as plt
import datetime


# Classical pre-screening (reduce 50 -> ≤33 assets)
def select_top_assets(mu, esg_s, top_k=33, method="return_esg"):
    """
    Simple classical feature reduction.
    - mu: expected returns
    - esg_s: ESG scores
    - method: scoring heuristic
    - top_k: number of assets to keep
    Returns the indices of the selected assets.
    """
    mu = np.array(mu)
    esg_s = np.array(esg_s)

    if method == "return_esg":
        # Example score: favor higher returns, moderate ESG
        scores = mu - 0.01 * esg_s
    elif method == "return_only":
        scores = mu
    elif method == "random":
        scores = np.random.random(len(mu))
    else:
        raise ValueError("Unknown selection method")

    # Pick top_k assets by score
    top_indices = np.argsort(scores)[-top_k:]
    return np.sort(top_indices)


def run_QAOA(qp, backend):
    
    cobyla = COBYLA()
    cobyla.set_options(maxiter=250)
    sampler = Sampler()
    sampler.set_options(backend=backend)
    qaoa_mes = QAOA(sampler=sampler, optimizer=cobyla, reps=3)
    qaoa = MinimumEigenOptimizer(qaoa_mes)
    return qaoa.solve(qp)

def get_subdic(dic, size=13):
    keys = list(dic.keys())
    selected_keys = np.random.choice(keys, size=size, replace=False)
    return {key: dic[key] for key in selected_keys}

def print_result(result, portfolio, print_comb=True):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    if print_comb:
        eigenstate = result.min_eigen_solver_result.eigenstate
        probabilities = (
            eigenstate.binary_probabilities()
            if isinstance(eigenstate, QuasiDistribution)
            else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
        )
        print("\n----------------- Full result ---------------------")
        print("selection\tvalue\t\tprobability")
        print("---------------------------------------------------")
        probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for k, v in probabilities:
            x = np.array([int(i) for i in list(reversed(k))])
            value = portfolio.to_quadratic_program().objective.evaluate(x)
            print("%10s\t%.4f\t\t%.4f" % (x, value, v))

    return selection

class PortfolioOptimization(OptimizationApplication):
    """Optimization application for the "portfolio optimization" [1] problem.

    References:
        [1]: "Portfolio optimization",
        https://en.wikipedia.org/wiki/Portfolio_optimization
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        covariances: np.ndarray,
        risk_factor: float,
        budget: int,
        esg_scores: Optional[np.ndarray] = None,
        roi_factor: Optional[float] = None,
        esg_factor: Optional[float] = None,
        bounds: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        expected_returns: 
            The expected returns for the assets.
        covariances: 
            The covariances between the assets.
        risk_factor: 
            The risk appetite of the decision maker.
        budget: 
            The budget, i.e. the number of assets to be selected.
        esg_scores: 
            The ESG scores for each asset. If None, ESG optimization is disabled.
        roi_factor: 
            The return appetite of the decision maker. If None, uses risk_factor for backward compatibility.
        esg_factor: 
            The ESG preference of the decision maker. If None, ESG optimization is disabled.
        bounds: 
            The list of tuples for the lower bounds and the upper bounds of each variable.
            e.g. [(lower bound1, upper bound1), (lower bound2, upper bound2), ...].
            Default is None which means all the variables are binary variables.
        """
        self._expected_returns = expected_returns
        self._covariances = covariances
        self._risk_factor = risk_factor
        self._budget = budget
        self._bounds = bounds
        # ESG-related parameters with backward compatibility
        self._esg_scores = esg_scores
        self._roi_factor = roi_factor if roi_factor is not None else risk_factor
        self._esg_factor = esg_factor if esg_factor is not None else 0.0
        self._check_compatibility()

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a portfolio optimization problem instance into a
        :class:`~qiskit_optimization.QuadraticProgram`.

        Returns:
            The :class:`~qiskit_optimization.QuadraticProgram` created
            from the portfolio optimization problem instance.
        """
        self._check_compatibility()
        num_assets = len(self._expected_returns)
        mdl = AdvModel(name="Portfolio optimization")
        
        if self._bounds: # More than one of each individual asset
            x = [mdl.integer_var(lb=self._bounds[i][0], ub=self._bounds[i][1], name=f"x_{i}") 
                 for i in range(num_assets)]
        else: # Only one of each individual function
            x = [mdl.binary_var(name=f"x_{i}") for i in range(num_assets)]
        
        # Build objective function components
        risk_term = mdl.quad_matrix_sum(self._covariances, x)
        return_term = np.dot(self._expected_returns, x)
        
        # Build the objective function with ESG term
        if self._esg_scores is not None and self._esg_factor != 0:
            esg_term = np.dot(self._esg_scores, x)
            objective = (self._risk_factor * risk_term - 
                        self._roi_factor * return_term - 
                        self._esg_factor * esg_term)
        else: # ESG term = 0
            objective = self._risk_factor * risk_term - return_term

        # Minimize objective function
        mdl.minimize(objective)
        mdl.add_constraint(mdl.sum(x[i] for i in range(num_assets)) == self._budget)
        op = from_docplex_mp(mdl)
        return op

    def portfolio_expected_value(self, result: Union[OptimizationResult, np.ndarray]) -> float:
        """Returns the portfolio expected value based on the result.

        Args:
            result: The calculated result of the problem

        Returns:
            The portfolio expected value
        """
        x = self._result_to_x(result)
        return np.dot(self._expected_returns, x)

    def portfolio_variance(self, result: Union[OptimizationResult, np.ndarray]) -> float:
        """Returns the portfolio variance based on the result

        Args:
            result: The calculated result of the problem

        Returns:
            The portfolio variance
        """
        x = self._result_to_x(result)
        return np.dot(x, np.dot(self._covariances, x))

    def portfolio_esg_score(self, result: Union[OptimizationResult, np.ndarray]) -> Optional[float]:
        """Returns the portfolio ESG score based on the result

        Args:
            result: The calculated result of the problem

        Returns:
            The portfolio ESG score, or None if ESG scores are not defined
        """
        if self._esg_scores is None:
            return None
        x = self._result_to_x(result)
        return np.dot(self._esg_scores, x)

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[int]:
        """Interpret a result as a list of asset indices

        Args:
            result: The calculated result of the problem

        Returns:
            The list of asset indices whose corresponding variable is 1
        """
        x = self._result_to_x(result)
        return [i for i, x_i in enumerate(x) if x_i]

    def _check_compatibility(self) -> None:
        """Check the compatibility of given variables"""
        if len(self._expected_returns) != len(self._covariances) or not all(
            len(self._expected_returns) == len(row) for row in self._covariances
        ):
            raise QiskitFinanceError(
                "The sizes of expected_returns and covariances do not match. ",
                f"expected_returns: {self._expected_returns}, covariances: {self._covariances}.",
            )
        
        # Check ESG scores compatibility if provided
        if self._esg_scores is not None:
            if len(self._expected_returns) != len(self._esg_scores):
                raise QiskitFinanceError(
                    "The sizes of expected_returns and esg_scores do not match. ",
                    f"expected_returns: {len(self._expected_returns)}, esg_scores: {len(self._esg_scores)}.",
                )
        
        if self._bounds is not None:
            if (
                not isinstance(self._bounds, list)
                or not all(isinstance(lb_, int) for lb_, _ in self._bounds)
                or not all(isinstance(ub_, int) for _, ub_ in self._bounds)
            ):
                raise QiskitFinanceError(
                    f"The bounds must be a list of tuples of integers. {self._bounds}",
                )
            if any(ub_ < lb_ for lb_, ub_ in self._bounds):
                raise QiskitFinanceError(
                    "The upper bound of each variable, in the list of bounds, must be greater ",
                    f"than or equal to the lower bound. {self._bounds}",
                )
            if len(self._bounds) != len(self._expected_returns):
                raise QiskitFinanceError(
                    f"The lengths of the bounds, {len(self._bounds)}, do not match to ",
                    f"the number of types of assets, {len(self._expected_returns)}.",
                )

    @property
    def expected_returns(self) -> np.ndarray:
        """Getter of expected_returns

        Returns:
            The expected returns for the assets.
        """
        return self._expected_returns

    @expected_returns.setter
    def expected_returns(self, expected_returns: np.ndarray) -> None:
        """Setter of expected_returns

        Args:
            expected_returns: The expected returns for the assets.
        """
        self._expected_returns = expected_returns

    @property
    def covariances(self) -> np.ndarray:
        """Getter of covariances

        Returns:
            The covariances between the assets.
        """
        return self._covariances

    @covariances.setter
    def covariances(self, covariances: np.ndarray) -> None:
        """Setter of covariances

        Args:
            covariances: The covariances between the assets.
        """
        self._covariances = covariances

    @property
    def risk_factor(self) -> float:
        """Getter of risk_factor

        Returns:
            The risk appetite of the decision maker.
        """
        return self._risk_factor

    @risk_factor.setter
    def risk_factor(self, risk_factor: float) -> None:
        """Setter of risk_factor

        Args:
            risk_factor: The risk appetite of the decision maker.
        """
        self._risk_factor = risk_factor

    @property
    def roi_factor(self) -> float:
        """Getter of roi_factor

        Returns:
            The return appetite of the decision maker.
        """
        return self._roi_factor

    @roi_factor.setter
    def roi_factor(self, roi_factor: float) -> None:
        """Setter of roi_factor

        Args:
            roi_factor: The return appetite of the decision maker.
        """
        self._roi_factor = roi_factor

    @property
    def esg_scores(self) -> Optional[np.ndarray]:
        """Getter of esg_scores

        Returns:
            The ESG scores for the assets.
        """
        return self._esg_scores

    @esg_scores.setter
    def esg_scores(self, esg_scores: np.ndarray) -> None:
        """Setter of esg_scores

        Args:
            esg_scores: The ESG scores for the assets.
        """
        self._esg_scores = esg_scores

    @property
    def esg_factor(self) -> float:
        """Getter of esg_factor

        Returns:
            The ESG preference of the decision maker.
        """
        return self._esg_factor

    @esg_factor.setter
    def esg_factor(self, esg_factor: float) -> None:
        """Setter of esg_factor

        Args:
            esg_factor: The ESG preference of the decision maker.
        """
        self._esg_factor = esg_factor

    @property
    def budget(self) -> int:
        """Getter of budget

        Returns:
            The budget, i.e. the number of assets to be selected.
        """
        return self._budget

    @budget.setter
    def budget(self, budget: int) -> None:
        """Setter of budget

        Args:
            budget: The budget, i.e. the number of assets to be selected.
        """
        self._budget = budget

    @property
    def bounds(self) -> List[Tuple[int, int]]:
        """Getter of the lower bounds and upper bounds of each selectable assets.

        Returns:
            The lower bounds and upper bounds of each assets selectable
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds: List[Tuple[int, int]]) -> None:
        """Setter of the lower bounds and upper bounds of each selectable assets.

        Args:
            bounds: The lower bounds and upper bounds of each assets selectable
        """
        self._bounds = bounds
        self._check_compatibility()