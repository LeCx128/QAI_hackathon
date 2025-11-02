from quantinuum_wrapper import QuantinuumWrapper
from pytket import Circuit
import matplotlib.pyplot as plt
import numpy as np

def run(input_data, roi_pct, risk_pct, esg_pct, available_qubits, solver_params, arguments):
    
    ##### THIW IS HOW YOU READ INPUT DATA FROM JSON #####
    
    #for asset in input_data:
    # print (asset, input_data[asset]['name']
    #data
    #######################################################
    # Parse JSON file
    from utils import Json_parse, get_subdic
    price_df, esg_s, asset_arr = Json_parse.json_parse(input_data)
    # price_df, esg_s, asset_arr = Json_parse.json_parse(get_subdic(dic, size=size))

    # Get mu & sigma
    from utils import Pre_processing
    mu, Sigma = Pre_processing.compute_mu_sigma(price_df)
    plt.imshow(Sigma, interpolation="nearest"); plt.show() # Sigma visualisation

    # # Simulate classical algorithm
    # from utils import run_portfolio_optimization_suite
    # config_example = {
    #     'compute_inputs': {'shrinkage': True},
    #     'scalarized_qp': {
    #         'lamb_grid': [0.5, 1.0, 2.0],
    #         'eta_grid': [0.0, 0.2, 0.5],
    #         'lb': 0.0, 'ub': 0.2, 'allow_short': False # Example: Max 20% per asset
    #     },
    #     'nsga2': {
    #         'pop_size': 150, 'generations': 80, 'plot': True
    #     },
    #     'cardinality': {
    #         'k': 15, 'lamb': 1.0, 'eta': 0.2
    #     },
    #     'cvar': {
    #         'alpha': 0.90, 'lamb': 0.5, 'eta': 0.3, 'use_mu_in_obj': True
    #     }
    # }
    # # Assume price_df and esg_df are loaded pandas DataFrames
    # results_dict = run_portfolio_optimization_suite(price_df, mu, Sigma, asset_arr, esg_s, config_example)
    # print("\n--- Results Summary ---")
    # print(results_dict.keys())


    # QAOA Portfolio Optimization
    from utils import PortfolioOptimization, print_result, run_QAOA, select_top_assets

    # Normalization
    roi_factor = roi_pct/risk_pct
    esg_factor = esg_pct/risk_pct

    # Solve qubit limit problem
    num_assets_total = len(mu)
    available_qubits = available_qubits
    
    # Select assets to fit hardware limit
    selected_idx = select_top_assets(mu, -esg_s, top_k=available_qubits)
    
    # Reduce problem data
    mu_red = np.array(mu)[selected_idx]
    Sigma_red = np.array(Sigma)[np.ix_(selected_idx, selected_idx)]
    esg_s_red = np.array(-esg_s)[selected_idx]

    # Parameters
    q = 1
    num_assets = len(mu)
    budget = num_assets // 2  
    portfolio = PortfolioOptimization(expected_returns=mu_red, covariances=Sigma_red, 
                                      risk_factor=q, budget=budget, 
                                      esg_scores=esg_s_red, roi_factor=roi_factor, esg_factor=0.01*esg_factor)
    qp = portfolio.to_quadratic_program()
    # Code for backend
    backend = QuantinuumWrapper.get_target()
    result = run_QAOA(qp, backend)
    q = print_result(result, portfolio)

    return {"result": str(q)}

    ##############################################
    ## THIS IS HOW YOU WILL ACCESS QUANTINUUM MACHINES ##
    
    # backend = QuantinuumWrapper.get_target()
    
    # circ = Circuit(2).H(0).CX(0, 1).CZ(0, 1)
    # circ.measure_all()
    # backend.default_compilation_pass().apply(circ)
    # compiled_circ = backend.get_compiled_circuit(circ)
    # handle = backend.process_circuit(compiled_circ, n_shots=3000)

    ## THE RETURN MUST BE JSON COMPATIBLE ##
    # return {"result": str(backend.get_result(handle).get_counts())}
