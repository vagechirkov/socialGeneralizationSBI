import json
import os
from datetime import datetime
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi.analysis import pairplot
from sbi.inference import NLE
from sbi.utils import BoxUniform, MultipleIndependent
from torch.distributions import Exponential, LogNormal
import torch.nn.functional as F

from model import model_sim, param_gen


def extract_history_vectors(df, n_options=121):
    df = df.sort_values(['group', 'round', 'agent', 'trial']).copy()

    # Add new empty columns for history vectors
    df['private_reward_vector'] = [[] for _ in range(len(df))]
    df['social_reward_vector'] = [[] for _ in range(len(df))]

    groups = df.groupby(['group', 'round', 'agent'])
    for (group, _round, agent), group_data in groups:
        for idx, row in group_data.iterrows():
            trial = int(row['trial'])

            # Private history: what this agent observed up to t-1
            private_vector = np.zeros(n_options)
            past_private = group_data.iloc[:trial]
            if not past_private.empty:
                for _, pr in past_private.iterrows():
                    choice = int(pr['choice'])
                    reward = pr['reward']
                    private_vector[choice] = reward

            # Social history: what others observed up to t-1
            social_vector = np.zeros(n_options)
            social_group = df[(df['group'] == group) & (df['round'] == _round) & (df['agent'] != agent) & (df['trial'] < trial)]
            if not social_group.empty:
                for _, sr in social_group.iterrows():
                    choice = int(sr['choice'])
                    reward = sr['reward']
                    if social_vector[choice] == 0:
                        social_vector[choice] = reward
                    else:
                        # If there are multiple observations, take the average
                        social_vector[choice] = (social_vector[choice] + reward) / 2

            df.at[idx, 'private_reward_vector'] = private_vector
            df.at[idx, 'social_reward_vector'] = social_vector

    return df


def prepare_theta_x_nle(simulations, params=('lambda', 'beta', 'tau', 'eps_soc'), n_options=121):
    theta_list = []
    x_list = []

    for idx, row in simulations.iterrows():
        # Extract parameter values
        param_values = [row[p] for p in params]

        # Extract private and social history vectors
        private_vector = row['private_reward_vector']
        social_vector = row['social_reward_vector']

        # Concatenate parameters and history into theta
        history_vector = np.concatenate([private_vector, social_vector])
        theta_row = np.concatenate([param_values])

        # One-hot encode choice
        choice = int(row['choice'])
        choice_one_hot = F.one_hot(torch.tensor(choice), num_classes=n_options).numpy()

        theta_list.append(theta_row)
        x_list.append(np.concatenate([choice_one_hot, history_vector]))

    theta = torch.tensor(np.array(theta_list), dtype=torch.float32)
    x = torch.tensor(np.array(x_list), dtype=torch.float32)

    return theta, x

def learn_likelihood(all_environments, save_dir, n_environments=5.0, n_groups_simulation=6000,
                     params=("lambda", "beta", "tau", "eps_soc"), n_options=121, n_rounds=8,
                     reward_min=-1, reward_max=1):
    params = list(params)
    _proposal = MultipleIndependent(
        [
            LogNormal(torch.tensor([-0.3]), torch.tensor([0.6])),  # lambda
            LogNormal(torch.tensor([-0.3]), torch.tensor([0.6])),  # beta
            LogNormal(torch.tensor([-4.0]), torch.tensor([0.6])),   # tau
            Exponential(torch.tensor([0.2])),                       # eps_soc

            # BoxUniform(torch.tensor([0.0001]), torch.tensor([2])),   # lambda
            # BoxUniform(torch.tensor([0.0001]), torch.tensor([2])),   # beta
            # BoxUniform(torch.tensor([0.0001]), torch.tensor([0.1])),   # tau
            # BoxUniform(torch.tensor([0.0001]), torch.tensor([20])),   # eps_soc

            # BoxUniform(torch.tensor([0.0]), torch.tensor([n_environments - 1.0])),   # env
            # BoxUniform(torch.full((n_options,), reward_min),
            #            torch.full((n_options,), reward_max)),  # private reward vector
            # BoxUniform(torch.full((n_options,), reward_min),
            #            torch.full((n_options,), reward_max)),  # social reward vector
        ],
        validate_args=False,
    )

    num_simulations = n_groups_simulation * 4
    proposal_samples = _proposal.sample((num_simulations,))

    # models=3 - social generalization
    pars = param_gen(4, n_groups_simulation, hom=True, models=3)

    for i in range(n_groups_simulation):
        for a in range(4):
            # set all parameters in pars to 0
            for k in pars[i][a].keys():
                pars[i][a][k] = 0.0

            for ii, p in enumerate(params):
                pars[i][a][p] = proposal_samples[i * 4 + a, ii].item()

    _used_envs = [[all_environments[i][ii] for ii in range(int(n_environments))] for i in range(len(all_environments))]

    simulations = model_sim(pars, _used_envs, n_rounds, 15, payoff=True)
    simulations = extract_history_vectors(simulations)
    # plt.plot(simulations.loc[14, 'private_reward_vector']); plt.show()
    # plt.plot(simulations.loc[14, 'social_reward_vector']); plt.show()

    # cols = params + ['env']
    # theta = torch.tensor(simulations[cols].to_numpy(), dtype=torch.float32)
    # x = torch.tensor(simulations['choice'].to_numpy(), dtype=torch.float32).unsqueeze(1)
    theta, x = prepare_theta_x_nle(simulations, params=params, n_options=n_options)

    trainer = NLE(_proposal, show_progress_bars=True, density_estimator="maf")
    estimator = trainer.append_simulations(theta, x)
    _inference = estimator.train()

    torch.save(_inference.state_dict(), save_dir / "inference.pth")
    return _inference

def load_train_likelihood(save_dir, n_environments):
    _proposal = MultipleIndependent(
        [
            BoxUniform(torch.tensor([0.0001]), torch.tensor([2])),   # lambda
            BoxUniform(torch.tensor([0.0001]), torch.tensor([2])),   # beta
            BoxUniform(torch.tensor([0.0001]), torch.tensor([0.1])),   # tau
            BoxUniform(torch.tensor([0.0001]), torch.tensor([20])),   # eps_soc
            # BoxUniform(torch.tensor([0.0]), torch.tensor([n_environments - 1.0])),   # env
            # BoxUniform(torch.full((n_options,), reward_min),
            #            torch.full((n_options,), reward_max)),  # private reward vector
            # BoxUniform(torch.full((n_options,), reward_min),
            #            torch.full((n_options,), reward_max)),  # social reward vector
        ],
        validate_args=False,
    )
    trainer = NLE(prior=_proposal, density_estimator="maf")
    _inference = trainer._build_neural_net()

    # Load weights
    _inference.load_state_dict(torch.load(save_dir / "inference.pth"))
    return _inference


def simulate_observations(all_environments, n_environments, n_groups_simulation=2,
                          ground_truth_params=(1.11, 0.33, 0.03, 12.55), params=("lambda", "beta", "tau", "eps_soc"),
                          n_rounds=8, simulated_agents=(0,)):
    params = list(params)
    sim_param_o = param_gen(4, n_groups_simulation, hom=True, models=3)
    _theta_o_df = pd.DataFrame(columns=params, index=[0], dtype=float)
    _theta_o_df.loc[0, :] = ground_truth_params

    # simulated data
    for i in range(n_groups_simulation):
        for a in simulated_agents:
            # set all parameters in pars to 0
            for k in sim_param_o[i][a].keys():
                sim_param_o[i][a][k] = 0.0

            for ii, p in enumerate(params):
                sim_param_o[i][a][p] = _theta_o_df.loc[0, p]
    _used_envs = [[all_environments[i][ii] for ii in range(int(n_environments))] for i in range(len(all_environments))]
    simulations_o = model_sim(sim_param_o, _used_envs, n_rounds, 15, payoff=True)
    simulations_o = extract_history_vectors(simulations_o)

    # theta_o = torch.tensor(simulations_o[params + ['env']].to_numpy(), dtype=torch.float32)
    # x_o = torch.tensor(simulations_o['choice'].to_numpy(), dtype=torch.float32).unsqueeze(1).T
    _theta_o, _x_o = prepare_theta_x_nle(simulations_o, params=params)
    return _theta_o, _theta_o_df, _x_o, simulations_o.agent.values


def human_observations():
    pass
    # return theta_o, theta_o_df, x_o


def fit_posterior(density_estimator, _theta_o, _theta_o_df, _x_o, _save_dir, num_samples=50_000,
                  params=("lambda", "beta", "tau", "eps_soc")):
    params = list(params)
    mcmc_parameters = dict(
        num_chains=7,
        thin=1,
        warmup_steps=400,
        init_strategy="proposal",
        num_workers=8
    )
    prior = MultipleIndependent(
        [
            # weakly informed priors
            LogNormal(torch.tensor([-0.3]), torch.tensor([0.6])),  # lambda
            LogNormal(torch.tensor([-0.3]), torch.tensor([0.6])),  # beta
            LogNormal(torch.tensor([-4.0]), torch.tensor([0.6])),   # tau
            Exponential(torch.tensor([0.2])),                       # eps_soc
        ],
        validate_args=False,
    )
    # prior_transform = mcmc_transform(prior)
    # potential_fn = LikelihoodBasedPotential(density_estimator, prior)
    #
    # conditioned_potential_fn = potential_fn.condition_on_theta(
    #     theta_o[:, 4:],  # pass only the conditions, must match the batch of iid data in x_o
    #     dims_global_theta=[0, 1, 2, 3]
    # )
    #
    # posterior = MCMCPosterior(
    #     # potential_fn=conditioned_potential_fn,
    #     theta_transform=prior_transform,
    #     proposal=prior,  # pass the prior, not the proposal.
    #     method="nuts_pyro",  # "nuts_pyro" "slice_np_vectorized"
    #     **mcmc_parameters
    # )
    # posterior_sample = posterior.sample((num_samples,), x=x_o.T)

    trainer = NLE(prior, density_estimator="maf")
    posterior = trainer.build_posterior(
        density_estimator=density_estimator,
        prior=prior,
        mcmc_method="nuts_pyro",  # "nuts_pyro" "slice_np_vectorized"
        mcmc_parameters=mcmc_parameters,
    )
    posterior_sample = posterior.sample((num_samples,), x=_x_o.T)

    # posterior = DirectPosterior(density_estimator, prior=prior)
    # posterior_sample = posterior.sample_batched((num_samples,), x=x_o.T)

    inference_data = posterior.get_arviz_inference_data()
    with az.style.context("arviz-darkgrid"):
        az.plot_trace(inference_data, compact=False)

    plt.savefig(_save_dir / "traceplot.png", dpi=300)
    plt.close()


    labels = ["$\lambda$", "$\\beta$", "$\\tau$", "$\\epsilon_{soc}$"]  #
    limits = [[0, 2], [0, 2], [0, 0.1], [-1, 20]]

    _ = pairplot(posterior_sample,
                 limits=limits,
                 points=np.array([_theta_o_df.loc[0, params]]),
                 figsize=(10, 10),
                 labels=labels,
                 )

    plt.savefig(_save_dir / "pairplot1.png", dpi=300)
    plt.close()


    limits = [[-0.1, 2.2], [-0.1, 2.2], [-0.01, 0.11], [-2, 22]]
    fig, ax = pairplot(
        [
            prior.sample((10_000,)),
            posterior_sample,
        ],
        points=np.array([_theta_o_df.loc[0, params]]),
        diag="kde",
        upper="contour",
        diag_kwargs=dict(bins=100),
        upper_kwargs=dict(levels=[0.95]),
        limits=limits,
        fig_kwargs=dict(
            points_offdiag=dict(marker="*", markersize=10),
            points_colors=["k"],
        ),
        labels=labels,
        figsize=(15, 15),
    )

    plt.sca(ax[1, 1])
    plt.legend(
        ["Prior", "Posterior", r"$\theta_o$"],
        frameon=False,
        fontsize=12,
    )
    plt.savefig(_save_dir / "pairplot2.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(42)

    save_dir = Path("parameter_recovery") / datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    path = 'data/environments'
    json_files = [file for file in os.listdir(path) if file.endswith('_canon.json')]
    all_environments = []
    for file in json_files:
        f=open(os.path.join(path, file))
        all_environments.append(json.load(f))

    inference = learn_likelihood(all_environments, save_dir, n_environments=1.0, n_groups_simulation=100, n_rounds=2)
    theta_o, theta_o_df, x_o, agent = simulate_observations(all_environments, n_environments=5, n_groups_simulation=2,
                                                     ground_truth_params=(1.11, 0.33, 0.03, 12.55))
    fit_posterior(inference, theta_o, theta_o_df, x_o.T, save_dir, num_samples=50_000)
    # npe_posterior_sample(inference, prior, theta_o_df, x_o, save_dir, num_samples=10)
