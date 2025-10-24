from quantinuum_wrapper import QuantinuumWrapper
from pytket import Circuit


def run(input_data, solver_params, extra_arguments):
    
    ##### THIW IS HOW YOU READ INPUT DATA FROM JSON #####
    
    #for asset in input_data:
    # print (asset, input_data[asset]['name']
    #data
    #######################################################


    ##############################################
    ## THIS IS HOW YOU WILL ACCESS QUANTINUUM MACHINES ##
    backend = QuantinuumWrapper.get_target()
    
    circ = Circuit(2).H(0).CX(0, 1).CZ(0, 1)
    circ.measure_all()
    backend.default_compilation_pass().apply(circ)
    compiled_circ = backend.get_compiled_circuit(circ)
    handle = backend.process_circuit(compiled_circ, n_shots=3000)

    ## THE RETURN MUST BE JSON COMPATIBLE ##
    return {"result": str(backend.get_result(handle).get_counts())}
