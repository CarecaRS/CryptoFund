#------ Mean-variance criterion (Inglese, L., 2022, Python for Finance and Algorithmic Trading, 2nd ed.)
def mvcriterion(df: pd.DataFrame, riskfree, weights, W=1, Lambda=3):
    import numpy as np
    """
    Inputs:
        df - Pandas DataFrame containing the historical data;
        riskfree - the risk-free rate *in percentage*;
        weights - weights for portfolio (ndarray);
        W - wealth of the portfolio (default 1);
        Lambda - level of risk aversion (default 3)

    Outputs: optimization portfolio criterion, in this structure
            the output is the inverse of the criterion, so as to
            minimize this in the portfolio optimization
    """

    # Adjust the risk free wealth (tax informed in percentage)
    Wbar = 1 + riskfree/100

    # Compute portfolio returns
    port_return = np.multiply(df, np.transpose(weights))
    port_return = port_return.sum(axis=1)

    # Compute the portfolio mean and volatility
    port_mean = np.mean(port_return, axis=0)
    port_std = np.std(port_return, axis=0)

    # Compute the criterion
    crit = ((Wbar ** (1-Lambda))/(1+Lambda)) + (Wbar**(-Lambda)) * W * port_mean - (Lambda/2) * Wbar **((-1-Lambda)*(W**2)*(port_std**2))
    crit = -crit

    return crit


#------ Portfolio optimization based on mvcriterion() (Inglese, L., 2022, Python for Finance and Algorithmic Trading, 2nd ed.)
def optim_mvcrit(df: pd.DataFrame):
    import numpy as np
    from scipy.optimize import minimize
    """
    Inputs:
        df - Pandas DataFrame containing the historical data;
    """

    # Estabilishes train/test sets
    cut = int(0.7*len(df))
    train_set = df.iloc[:cut, :]
    test_set = df.iloc[cut:, :]

    # Number of assets
    num_assets = df.shape[1]

    # Initialization weight value
    x0 = np.ones(num_assets)

    # Optimization constraints
    constraints = (['type': 'eq', 'fun': lambda x: sum(abs(x)) -1])

    # Boundaries
    bounds = [(0,1) for i in range (0, num_assets)]

    # Solving optimization
    res_MV = minimize(mvcriterion, x0,
                      method='SLSQP',
                      args=(train_set),
                      bounds=bounds,
                      constraints=constraints,
                      options=['disp':True])
    
    # Result
    return res_MV.x


#------ Value at Risk (VaR) (Inglese, L., 2022, Python for Finance and Algorithmic Trading, 2nd ed.)
def VaR():
    pass


