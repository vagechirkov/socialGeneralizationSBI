"""
Script Name: model.py
Authors: A. Witt, W. Toyokawa, K.N. Lala, W. Gaissmaier, & C.M. Wu,
Copyright (c) 2019-2024 Charley Wu et al.
Licensed under the MIT License (see LICENSE file for details)

Associated Publication:
A. Witt, W. Toyokawa, K.N. Lala, W. Gaissmaier, & C.M. Wu,
"Humans flexibly integrate social information despite interindividual differences in reward,"
Proc. Natl. Acad. Sci. U.S.A. 121 (39) e2404928121, (2024).
https://doi.org/10.1073/pnas.2404928121

If you use this software in your research, please cite the above publication.

Modifications:
- Valerii Chirkov, 2025: adjusted the code to run simulation-based inference of model parameters
"""

# MIT License
#
# Copyright (c) 2019 Charley Wu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def ucb(pred, beta=0.5):
    """
    Parameters
    ----------
    pred : tuple of arrays in shape (2,n,1)
        mean and variance values for n options (GPR prediction output).
    beta : numerical, optional
        governs an exploration tendency. The default is 0.5.

    Returns
    -------
    Upper confidence bound of the data.
    """
    out = pred[0] + beta * np.sqrt(pred[1])
    return (out)


def rbf(x, y, _l):
    D = cdist(x, y) ** 2
    return (np.exp(-D / (2 * _l ** 2)))


def GPR(Xstar, obs, rewards, socIndex, pars, baseEpsilon, k=rbf):
    """
    Parameters
    ----------
    Xstar : list
        The space we project to. Maps observation indices to coordinates as well.
    obs : list
        Indices of agent's observation history.
    rewards : list
        Agent's reward history.
    socIndex : list
        Relevant for SG model; 1 if observation is social, 0 if individual.
    pars : dict        Model parameters.
    k : function, optional
        Kernel function. The default is rbf.

    Returns
    -------
    List of posterior mean and variance.

    """
    choices = np.array([Xstar[x] for x in obs])
    K = k(choices, choices, pars['lambda'])  # covariance of observations
    # add noise (social noise gets added to social observations)
    noise = np.ones_like(socIndex) * baseEpsilon + socIndex * pars['eps_soc']
    KK = K + np.identity(len(noise)) * noise
    KK_inv = np.linalg.inv(KK)
    Ky = np.dot(KK_inv, rewards)
    Kstarstar = k(Xstar, Xstar, pars['lambda'])  # covariance of Xstar with itself
    Kstar = k(Xstar, choices, pars['lambda'])
    mu = np.dot(Kstar, Ky)  # get mean
    cov = Kstarstar - np.dot(np.dot(Kstar, KK_inv), Kstar.T)  # covariance
    var = np.diagonal(cov).copy()  # and variance; if I don't copy, var isn't writeable, which breaks for SG
    return ([mu, var])


def param_gen_resample_lam(nAgents, nGroups, betList, tauList, epsList, models=None):
    """
    Generates parameter sets for simulations

    Parameters
    ----------
    nAgents : int
        Number of agents per group. Since we have 4 environments per set, the default is 4.
    nGroups : int
        Number of groups.
    hom : boolean, optional
        Should the groups be homogenous. The default is True.
    models : int in range(5), optional
        If specified, homogenous groups will only have the input model.
        0=AS,1=DB,2=VS,3=SG,4=dummy model. The default is None.

    Returns
    -------
    List of list of dictionaries as used in model_sim.

    """
    par_names = ["lambda", "beta", "tau", "gamma", "alpha", "eps_soc", "dummy"]  # needed for dicts later #
    all_pars = []
    # randomly select model unless given as an argument
    if models is None:
        models = np.random.randint(0, 2, nGroups)
    else:
        assert models in range(0, 5), "Model has to be between 0 and 4"
        models = np.ones(nGroups) * models
    for g in range(nGroups):
        model = models[g]
        all_pars.append([])  # set up list for parameter dictionary
        par = np.zeros((nAgents, len(par_names)))
        par[:, 0] = np.random.uniform(0.5, 2.5, (nAgents))  # lambda
        par[:, 1] = np.random.choice(betList, nAgents)
        # par[:,1] = np.random.lognormal(-0.75,0.5,(nAgents))
        par[:, 2] = np.random.choice(tauList, nAgents)  # tau
        if model == 1:
            # par[:,5] = np.random.uniform(0,19,(nAgents)) #eps_soc
            par[:, 5] = np.random.choice(epsList, nAgents)
        for ag in range(nAgents):
            pars = dict(zip(par_names, par[ag, :]))
            all_pars[g].append(pars)
    return (all_pars)


def param_gen(nAgents, nGroups, hom=True, models=None):
    """
    Generates parameter sets for simulations

    Parameters
    ----------
    nAgents : int
        Number of agents per group. Since we have 4 environments per set, the default is 4.
    nGroups : int
        Number of groups.
    hom : boolean, optional
        Should the groups be homogenous. The default is True.
    models : int in range(5), optional
        If specified, homogenous groups will only have the input model.
        0=AS,1=DB,2=VS,3=SG,4=dummy model. The default is None.

    Returns
    -------
    List of list of dictionaries as used in model_sim.

    """
    par_names = ["lambda", "beta", "tau", "gamma", "alpha", "eps_soc", "dummy"]  # needed for dicts later #
    all_pars = []
    # homogenous groups
    if hom:
        # randomly select model unless given as an argument
        if models is None:
            models = np.random.randint(0, 4, nGroups)
        else:
            assert models in range(0, 5), "Model has to be between 0 and 4"
            models = np.ones(nGroups) * models
        for g in range(nGroups):
            model = models[g]
            all_pars.append([])  # set up list for parameter dictionary
            par = np.zeros((nAgents, len(par_names)))
            par[:, 0] = np.random.lognormal(-0.75, 0.5, (nAgents))  # lambda
            par[:, 1] = np.random.lognormal(-0.75, 0.5, (nAgents))
            par[:, 2] = np.random.lognormal(-4.5, 0.9, (nAgents))  # tau
            if model == 1:
                par[:, 3] = np.random.uniform(0, 1, (nAgents))  # gamma previously 1/14
            elif model == 2:
                par[:, 4] = np.random.uniform(0, 1, (nAgents))
            elif model == 3:
                par[:, 5] = np.random.exponential(2, (nAgents))  # eps_soc
            elif model == 4:
                par[:, 6] = np.ones(nAgents)  # dummy variable for a model test
            for ag in range(nAgents):
                pars = dict(zip(par_names, par[ag, :]))
                all_pars[g].append(pars)
    else:  # heterogeneous groups
        for g in range(nGroups):
            all_pars.append([])
            model = np.random.randint(0, 4, (nAgents))
            par = np.zeros((nAgents, len(par_names)))
            par[:, 0] = np.random.lognormal(-0.75, 0.5, (nAgents))  # lambda
            par[:, 1] = np.random.lognormal(-0.75, 0.5, (nAgents))  # beta
            par[:, 2] = np.random.lognormal(-4.5, 0.9, (nAgents))  # tau
            for ag in range(nAgents):
                if model[ag] == 1:
                    par[ag, 3] = np.random.uniform(0.2142857, 1, (1))  # gamma
                elif model[ag] == 2:
                    par[ag, 4] = np.random.uniform(0.116, 1, (1))
                elif model[ag] == 3:
                    while par[ag, 5] == 0 or par[ag, 5] > 19:
                        par[ag, 5] = np.random.exponential(2, (1))  # eps_soc
                elif model[ag] == 4:
                    par[ag, 6] = 1  # dummy variable for a model test
                pars = dict(zip(par_names, par[ag, :]))
                all_pars[g].append(pars)
    return (all_pars)


def model_sim_orig(allParams, envList, rounds, shor, baseEpsilon=.0001, payoff=True, prior_mean=0.5, prior_scale=1):
    """
    Simulates experimental data based on input specifications.
    This is the original version of the function to compare with the updated version.

    Parameters
    ----------
    allParams : list of list of dictionaries
        List of agent's parameter values. Number of groups is specified based on this.
    envList : list
        Reward environments.
    rounds : int
        Number of rounds (and thus environments used).
    shor : int
        Search horizon (number of datapoints per round).
    baseEpsilon : numeric, optional
        Assumed observation noise for individual observations. The default is .0001.
    payoff : Boolean, optional
        Should VS use outcome information. The default is True.

    Returns
    -------
    A pandas dataframe with the simulated data (specific agents, environments, scores and parameters).
    """
    # set up lists to collect output info
    agent = []
    group = []
    r0und = []
    env = []
    trial = []
    choice = []
    reward = []
    lambda_ind = []
    dummy = []
    beta = []
    tau = []
    gamma = []
    alpha = []
    eps_soc = []
    # simulation parameters
    nAgents = len(allParams[0])
    gridSize = len(envList[0][0])
    Xstar = np.array(
        [(x, y) for x in range(np.sqrt(gridSize).astype(int)) for y in range(np.sqrt(gridSize).astype(int))])

    for g in range(len(allParams)):  # iterate over groups
        # get parameters and set up collectors for value and policy
        pars = allParams[g]
        vals = np.zeros((gridSize, nAgents))
        policy = vals = np.zeros((gridSize, nAgents))
        # sample random set of environments
        envs = np.random.randint(0, len(envList[0]), (rounds))  # change here to use set envs
        for r in range(rounds):
            # in every round, reset observations and rewards
            X = np.zeros((shor, nAgents)).astype(int)
            Y = np.zeros((shor, nAgents))
            for t in range(shor):
                # on each trial, reset prediction
                prediction = []
                if t == 0:
                    # First choice is random
                    X[0, :] = np.random.randint(0, gridSize, (nAgents))
                    Y[0, :] = [
                        (envList[ag][envs[r]][X[0, ag]]['payoff'] - prior_mean) / prior_scale + np.random.normal(0, .01)
                        for ag in range(nAgents)]
                else:
                    for ag in range(nAgents):
                        obs = X[:t, ag]  # self observations
                        rewards = Y[:t, ag]  # self rewards
                        socIndex = np.zeros_like(obs)  # individual info
                        # social generalization uses social obs in GP
                        if (pars[ag]['eps_soc'] > 0) or (pars[ag]["dummy"] != 0):
                            obs = np.append(obs, X[:t, np.arange(nAgents) != ag])  # social observations
                            rewards = np.append(rewards, Y[:t, np.arange(nAgents) != ag])  # social rewards
                            if pars[ag]["dummy"] != 0:
                                socIndex = np.zeros_like(obs)  # dummy model treats everything the same
                            else:
                                socIndex = np.append(socIndex, np.ones_like(
                                    X[:t, np.arange(nAgents) != ag]))  # otherwise flag as social observations

                        prediction.append(GPR(Xstar, obs, rewards, socIndex, pars[ag], baseEpsilon))
                        # values from UCB
                        vals[:, ag] = np.squeeze(ucb(prediction[ag], pars[ag]['beta']))
                        # count occurrences of social choices in the previous round
                        bias = [(i, np.sum(X[t - 1, np.arange(nAgents) != ag] == i),
                                 np.mean(Y[t - 1, np.arange(nAgents) != ag][
                                             np.nonzero(X[t - 1, np.arange(nAgents) != ag] == i)]))
                                # mean of Y at t-1 for each unique X at t-1
                                for i in np.unique(X[t - 1, np.arange(nAgents) != ag])]

                        # Value shaping
                        if pars[ag]['alpha'] > 0:  # prediction error learning from individual value vs. soc info
                            vs_boost = [(i, np.sum(X[:t, np.arange(nAgents) != ag] == i), np.mean(Y[X == i])) for i in
                                        np.unique(X[:t, np.arange(nAgents) != ag])]
                            # vs_boost = [(X[np.where(Y==i)],i) for i in np.max(Y[:t,np.arange(nAgents)!=ag],axis=0)]
                            for b in vs_boost:
                                # vals[b[0],ag] += pars[ag]['alpha']*b[1]*(b[2]-rew_exp)
                                vals[b[0], ag] = vals[b[0], ag] + pars[ag]['alpha'] * (
                                        b[1] * b[2] - vals[b[0], ag])  # np.mean(rewards)

                        # avoid overflow
                        vals[:, ag] = vals[:, ag] - np.max(vals[:, ag])
                        # Policy
                        policy[:, ag] = np.exp(vals[:, ag] / pars[ag]['tau'])
                        # Decision biasing
                        if pars[ag]['gamma'] > 0:
                            # payoff bias (generally unused)
                            rew_exp = np.mean(rewards)  # individual reward experience
                            socChoices = np.ones(gridSize) * .000000000001  # just zeros breaks policy sometimes
                            if payoff:
                                for b in bias:
                                    # subtracting from policy overcomplicates things; only boost helpful info
                                    if b[2] - rew_exp < 0:
                                        continue
                                    # social policy proportional to how much better than experience social option is
                                    socChoices[b[0]] += b[1] * (b[2] - rew_exp)
                            else:
                                for b in bias:
                                    socChoices[b[0]] += b[1]
                            socpolicy = socChoices / sum(socChoices)
                            # mixture policy
                            policy[:, ag] = ((1 - pars[ag]['gamma']) * policy[:, ag]) + (pars[ag]['gamma'] * socpolicy)
                    # Choice
                    policy = policy / np.sum(policy, axis=0)
                    X[t, :] = [np.random.choice(gridSize, p=policy[:, ag]) for ag in range(nAgents)]
                    Y[t, :] = [
                        (envList[ag][envs[r]][X[t, ag]]['payoff'] - prior_mean) / prior_scale + np.random.normal(0, .01)
                        for ag in range(nAgents)]
            for ag in range(nAgents):
                # collect information
                agent.append(np.ones((shor, 1)) * ag)
                group.append(np.ones((shor, 1)) * g)
                r0und.append(np.ones((shor, 1)) * r)
                env.append(np.ones((shor, 1)) * envs[r])
                trial.append(np.arange(shor))
                choice.append(X[:, ag])
                reward.append(Y[:, ag])
                lambda_ind.append(np.ones((shor, 1)) * pars[ag]['lambda'])
                beta.append(np.ones((shor, 1)) * pars[ag]['beta'])
                tau.append(np.ones((shor, 1)) * pars[ag]['tau'])
                gamma.append(np.ones((shor, 1)) * pars[ag]['gamma'])
                alpha.append(np.ones((shor, 1)) * pars[ag]['alpha'])
                eps_soc.append(np.ones((shor, 1)) * pars[ag]['eps_soc'])
                dummy.append(np.ones((shor, 1)) * pars[ag]["dummy"])
    # format dataset
    data = np.column_stack((np.concatenate(agent), np.concatenate(group), np.concatenate(r0und), np.concatenate(env),
                            np.concatenate(trial),
                            np.concatenate(choice), np.concatenate(reward), np.concatenate(lambda_ind),
                            np.concatenate(beta), np.concatenate(tau), np.concatenate(gamma), np.concatenate(alpha),
                            np.concatenate(eps_soc), np.concatenate(dummy)))
    agentData = pd.DataFrame(data,
                             columns=('agent', 'group', 'round', 'env', 'trial', 'choice', 'reward', 'lambda', 'beta',
                                      'tau', 'gamma', 'alpha', 'eps_soc', "dummy"))
    return (agentData)


def model_sim(allParams, envList, rounds, shor, baseEpsilon=.0001, payoff=True, prior_mean=0.5, prior_scale=1):
    """
    Simulates experimental data based on input specifications

    Parameters
    ----------
    allParams : list of list of dictionaries
        List of agent's parameter values. Number of groups is specified based on this.
    envList : list
        Reward environments.
    rounds : int
        Number of rounds (and thus environments used).
    shor : int
        Search horizon (number of datapoints per round).
    baseEpsilon : numeric, optional
        Assumed observation noise for individual observations. The default is .0001.
    payoff : Boolean, optional
        Should VS use outcome information. The default is True.

    Returns
    -------
    A pandas dataframe with the simulated data (specific agents, environments, scores and parameters).
    """
    # set up lists to collect output info
    data_columns = ['agent', 'group', 'round', 'env', 'trial', 'choice', 'reward', 'lambda', 'dummy', 'beta', 'tau',
                    'gamma', 'alpha', 'eps_soc']
    data_dict = {col: [] for col in data_columns}

    # simulation parameters
    nAgents = len(allParams[0])
    gridSize = len(envList[0][0])
    Xstar = np.array(
        [(x, y) for x in range(np.sqrt(gridSize).astype(int)) for y in range(np.sqrt(gridSize).astype(int))])

    for g, group_params in enumerate(allParams):  # iterate over groups
        group_data = simulate_group(group_params, envList, rounds, shor, baseEpsilon, payoff, prior_mean, prior_scale,
                                    Xstar, nAgents, gridSize, g)
        for col in data_columns:
            data_dict[col].extend(group_data[col])

    # format dataset
    data = np.column_stack([np.concatenate(data_dict[col]) for col in data_columns])
    agentData = pd.DataFrame(data, columns=data_columns)
    return agentData


def simulate_group(pars, envList, rounds, shor, baseEpsilon, payoff, prior_mean, prior_scale, Xstar, nAgents, gridSize,
                   group_id):
    """
    Simulates data for a single group of agents.

    Parameters
    ----------
    pars : list of dictionaries
        List of agent's parameter values for the group.
    envList : list
        Reward environments.
    rounds : int
        Number of rounds (and thus environments used).
    shor : int
        Search horizon (number of datapoints per round).
    baseEpsilon : numeric
        Assumed observation noise for individual observations.
    payoff : Boolean
        Should VS use outcome information.
    prior_mean : float
        Prior mean for normalization.
    prior_scale : float
        Prior scale for normalization.
    Xstar : numpy array
        Array of coordinates for the grid.
    nAgents : int
        Number of agents.
    gridSize : int
        Size of the grid.
    group_id : int
        ID of the group.

    Returns
    -------
    dict
        Dictionary containing the simulated data for the group.
    """
    data = {key: [] for key in
            ['agent', 'group', 'round', 'env', 'trial', 'choice', 'reward', 'lambda', 'dummy', 'beta', 'tau',
             'gamma', 'alpha', 'eps_soc']}
    policy = np.zeros((gridSize, nAgents))
    vals = np.zeros((gridSize, nAgents))
    envs = np.random.randint(0, len(envList[0]), rounds)

    for r in range(rounds):
        X, Y = np.zeros((shor, nAgents), dtype=int), np.zeros((shor, nAgents))
        for t in range(shor):
            if t == 0:
                X[0, :] = np.random.randint(0, gridSize, nAgents)
                Y[0, :] = _compute_reward_all_agents(envList, envs, X, t, r, nAgents, prior_mean, prior_scale)
            else:
                for ag in range(nAgents):
                    obs, rewards = X[:t, ag], Y[:t, ag]
                    socIndex = np.zeros_like(obs)
                    if pars[ag]['eps_soc'] > 0 or pars[ag]["dummy"] != 0:
                        obs, rewards, socIndex = _social_generalization_model(X, Y, ag, nAgents, obs, pars, rewards,
                                                                              socIndex, t)

                    prediction = GPR(Xstar, obs, rewards, socIndex, pars[ag], baseEpsilon)
                    vals[:, ag] = np.squeeze(ucb(prediction, pars[ag]['beta']))

                    if pars[ag]['alpha'] > 0:
                        _value_shaping_model(X, Y, ag, nAgents, pars, t, vals)

                    vals[:, ag] -= np.max(vals[:, ag])
                    policy[:, ag] = np.exp(vals[:, ag] / pars[ag]['tau'])

                    if pars[ag]['gamma'] > 0:
                        policy = _decision_biasing_model(X, Y, ag, gridSize, nAgents, pars, payoff, policy, rewards, t)

                policy /= np.sum(policy, axis=0)
                X[t, :] = [np.random.choice(gridSize, p=policy[:, ag]) for ag in range(nAgents)]
                Y[t, :] = _compute_reward_all_agents(envList, envs, X, t, r, nAgents, prior_mean, prior_scale)

        for ag in range(nAgents):
            for key, value in zip(data.keys(),
                                  [ag, group_id, r, envs[r], np.arange(shor), X[:, ag], Y[:, ag], pars[ag]['lambda'],
                                   pars[ag]['dummy'], pars[ag]['beta'], pars[ag]['tau'], pars[ag]['gamma'],
                                   pars[ag]['alpha'], pars[ag]['eps_soc']]):
                data[key].append(
                    np.ones((shor, 1)) * value if key in ['agent', 'group', 'round', 'env', 'lambda', 'dummy',
                                                          'beta', 'tau', 'gamma', 'alpha', 'eps_soc'] else value)
    return data


def _compute_reward_all_agents(_envs, env_ids, X, t, r, nAgents, prior_mean, prior_scale):
    return [_compute_reward(_envs[a][env_ids[r]][X[t, a]]['payoff'], prior_mean, prior_scale) for a in range(nAgents)]


def _compute_reward(payoff, prior_mean, prior_scale):
    return _normalize_by_prior(payoff, prior_mean, prior_scale) + np.random.normal(0, .01)


def _normalize_by_prior(Y, prior_mean, prior_scale):
    return (Y - prior_mean) / prior_scale


def _decision_biasing_model(X, Y, ag, gridSize, nAgents, pars, payoff, policy, rewards, t):
    bias = [(i, np.sum(X[t - 1, np.arange(nAgents) != ag] == i),
             np.mean(Y[t - 1, np.arange(nAgents) != ag][np.nonzero(X[t - 1, np.arange(nAgents) != ag] == i)])) for i in
            np.unique(X[t - 1, np.arange(nAgents) != ag])]
    rew_exp = np.mean(rewards)
    socChoices = np.ones(gridSize) * 1e-12
    for b in bias:
        if payoff and b[2] - rew_exp >= 0:
            socChoices[b[0]] += b[1] * (b[2] - rew_exp)
        elif not payoff:
            socChoices[b[0]] += b[1]
    socpolicy = socChoices / sum(socChoices)
    policy[:, ag] = (1 - pars[ag]['gamma']) * policy[:, ag] + pars[ag]['gamma'] * socpolicy
    return policy


def _social_generalization_model(X, Y, ag, nAgents, obs, pars, rewards, socIndex, t):
    obs = np.append(obs, X[:t, np.arange(nAgents) != ag])
    rewards = np.append(rewards, Y[:t, np.arange(nAgents) != ag])
    socIndex = np.zeros_like(obs) if pars[ag]["dummy"] != 0 else np.append(socIndex, np.ones_like(
        X[:t, np.arange(nAgents) != ag]))
    return obs, rewards, socIndex


def _value_shaping_model(X, Y, ag, nAgents, pars, t, vals):
    vs_boost = [(i, np.sum(X[:t, np.arange(nAgents) != ag] == i), np.mean(Y[X == i])) for i in
                np.unique(X[:t, np.arange(nAgents) != ag])]
    for b in vs_boost:
        vals[b[0], ag] += pars[ag]['alpha'] * (b[1] * b[2] - vals[b[0], ag])


if __name__ == "__main__":
    import json
    import os

    path = 'data/environments'
    json_files = [file for file in os.listdir(path) if file.endswith('_canon.json')]
    envList = []
    for file in json_files:
        f=open(os.path.join(path, file))
        envList.append(json.load(f))

    for m in range(5):
        # pars_lam = param_gen_resample_lam(4, 3, betList, tauList, epsList, models=m)
        pars = param_gen(4, 3, hom=True, models=m)

        # set the seed for reproducibility
        np.random.seed(42)
        c_orig = model_sim_orig(pars, envList, 8, 15, payoff=True)

        np.random.seed(42)
        c = model_sim(pars, envList, 8, 15, payoff=True)

        # check if the original and updated versions of the function produce the same output
        print(c_orig.sort_index(axis=1).equals(c.sort_index(axis=1)))