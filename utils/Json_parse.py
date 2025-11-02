# Imports
import numpy as np
import pandas as pd

def get_subdic(dic, size=13):
    keys = list(dic.keys())
    selected_keys = np.random.choice(keys, size=size, replace=False)
    return {key: dic[key] for key in selected_keys}

def json_parse(input_data):
    """
    Reads Json data and outputs price and esg score of each assets as well as an array of all asset ID

    Parameters
    ----------
    input_data : dict
        Dictionary from input_data.json after json.load()

    Returns
    ----------
    price_df : pd.DataFrame
        DataFrame of price data with index as dates and columns as asset IDs
    esg_s : pd.Series(esg_dict)
        Series of esgScore data with index as assetIDs
    asset_arr : np.array
        Array of all asset IDs
    """
    asset_arr = np.empty(len(input_data),dtype=object)

    iterator = 0
    price_dict = {}
    esg_dict = {}
    for asset_code in input_data:
        asset_arr[iterator] = asset_code

        # price_dict
        history_dict = input_data[asset_code]["History"]
        filtered_history_dict = {date:history_dict[date]["Close"] for date in history_dict} # Use Close price as we assume "hold overnight" investors

        price_dict[asset_code] = filtered_history_dict

        # esg_dict
        esgScores_dict = input_data[asset_code]["Sustainability"]["esgScores"]
        environmentScore = esgScores_dict["environmentScore"]
        governanceScore = esgScores_dict["governanceScore"]
        socialScore = esgScores_dict["socialScore"]
        avg_esgScore = (environmentScore + governanceScore + socialScore)/3 # Take an average
        esg_dict[asset_code] = avg_esgScore
        
        iterator += 1

    
    price_df = pd.DataFrame.from_dict(price_dict)
    esg_s = pd.Series(esg_dict)

    return price_df, esg_s, asset_arr