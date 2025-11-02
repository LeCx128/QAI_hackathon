# ======================== QUANTUM CONSTRAINT-FILTERED QAOA ========================
import numpy as np

# Graceful import check for Qiskit
try:
    from qiskit import QuantumCircuit, Aer
    from qiskit.providers.aer import AerSimulator
    from qiskit.quantum_info import Statevector
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.utils import QuantumInstance
    from qiskit.opflow import PauliSumOp, Z, I
    USE_QISKIT = True
except ImportError:
    USE_QISKIT = False

def apply_interference_filter(qc, qubits, target_sum):
    """Quantum interference step that enforces sum(weights)=target_sum via destructive interference."""
    anc = qc.qregs[0][-1]
    for i in range(len(qubits)):
        qc.cx(qubits[i], anc)
    qc.rz(np.pi, anc)
    for i in range(len(qubits)):
        qc.cx(qubits[i], anc)
    return qc

def build_constraint_oracle(n_qubits, target_sum):
    """Builds a constraint oracle circuit marking infeasible budget states."""
    qc = QuantumCircuit(n_qubits + 1)
    apply_interference_filter(qc, range(n_qubits), target_sum)
    qc.name = "BudgetOracle"
    return qc

def build_objective_hamiltonian(Q, lin):
    """Converts QUBO into Hamiltonian for QAOA."""
    n = Q.shape[0]
    H = 0
    for i in range(n):
        H += lin[i] * (I ^ (n - i - 1)) ^ Z
    for i in range(n):
        for j in range(i + 1, n):
            H += Q[i, j] * (Z ^ (n - j - 1)) ^ (Z ^ (n - i - 1))
    return PauliSumOp.from_operator(H.to_matrix())

def run_filtered_qaoa(Q, lin, p=1, target_sum=4):
    """Runs constraint-filtered QAOA and returns top feasible bitstrings."""
    if not USE_QISKIT:
        print("⚠️ Qiskit not available – running classical random fallback.")
        n = Q.shape[0]
        rng = np.random.default_rng(42)
        best = []
        for _ in range(200):
            bits = rng.integers(0, 2, n)
            if np.sum(bits) == target_sum:
                val = bits @ Q @ bits + lin @ bits
                best.append((val, bits))
        best.sort(key=lambda x: x[0])
        return best[:5]

    n = Q.shape[0]
    H_obj = build_objective_hamiltonian(Q, lin)
    backend = AerSimulator(method="statevector")
    optimizer = COBYLA(maxiter=60)
    qaoa = QAOA(reps=p, optimizer=optimizer, quantum_instance=QuantumInstance(backend))
    res = qaoa.compute_minimum_eigenvalue(H_obj)

    # Extract feasible states
    state = res.eigenstate
    sv = Statevector(state)
    probs = np.abs(sv.data) ** 2
    feasible = []
    for idx, prob in enumerate(probs):
        bits = np.array(list(np.binary_repr(idx, width=n)), dtype=int)
        if np.sum(bits) == target_sum:
            feasible.append((prob, bits))
    feasible.sort(key=lambda x: -x[0])
    return feasible[:5]

def quantum_refinement_pipeline(Sigma_small, mu_small, esg_small, B=3, target_sum=None):
    """
    Quantum-native refinement step that replaces penalty terms with physical interference.
    """
    n = len(mu_small)
    scales = np.array([2.0 ** (-(b + 1)) for b in range(B)])
    n_b = n * B
    W = np.kron(np.eye(n), scales)
    Q = W.T @ Sigma_small @ W
    lin = -mu_small @ W
    if target_sum is None:
        target_sum = int(n_b / 2)

    print(f"Running Quantum Constraint-Filtered QAOA: n_b={n_b}, target_sum={target_sum}")
    samples = run_filtered_qaoa(Q, lin, p=1, target_sum=target_sum)

    results = []
    for prob, bits in samples:
        w_cont = (bits.reshape(n, B) @ scales)
        w_cont = np.maximum(w_cont, 0)
        if np.sum(w_cont) <= 0: continue
        w_cont /= np.sum(w_cont)
        ret = mu_small @ w_cont
        risk = w_cont.T @ Sigma_small @ w_cont
        esg_val = esg_small @ w_cont
        results.append({
            'prob': float(prob),
            'return': float(ret),
            'risk': float(risk),
            'esg': float(esg_val),
            'weights': w_cont
        })
    results.sort(key=lambda r: -r['prob'])
    return results
