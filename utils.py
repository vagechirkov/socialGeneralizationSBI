import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi.analysis import pairplot
from sbi.inference import SNPE
from sbi.utils import BoxUniform, MultipleIndependent
from torch.distributions import Exponential, LogNormal
import torch.nn.functional as F

from model import model_sim, param_gen

box_uniform_prior = MultipleIndependent(
    [
        BoxUniform(torch.tensor([0.0001]), torch.tensor([2])),   # lambda
        BoxUniform(torch.tensor([0.0001]), torch.tensor([2])),   # beta
        BoxUniform(torch.tensor([0.0001]), torch.tensor([0.1])),   # tau
        BoxUniform(torch.tensor([0.0001]), torch.tensor([20])),   # eps_soc
    ],
    validate_args=False,
)

weakly_informed_prior = MultipleIndependent(
    [
        LogNormal(torch.tensor([-0.3]), torch.tensor([0.6])),  # lambda
        LogNormal(torch.tensor([-0.3]), torch.tensor([0.6])),  # beta
        LogNormal(torch.tensor([-4.0]), torch.tensor([0.6])),   # tau
        Exponential(torch.tensor([0.2])),                       # eps_soc
    ],
    validate_args=False,
)



def extract_history_vectors(df, n_options=121):
    df = df.sort_values(['group', 'round', 'agent', 'trial']).reset_index(drop=True)

    # Prepare columns to hold vectors as arrays
    df['private_reward_vector'] = [np.zeros(n_options) for _ in range(len(df))]
    df['social_reward_vector'] = [np.zeros(n_options) for _ in range(len(df))]

    # Precompute group-round-agent and group-round splits to avoid repeated filtering
    group_round_agent_groups = df.groupby(['group', 'round', 'agent'], sort=False)
    group_round_groups = df.groupby(['group', 'round'], sort=False)

    # Store per-agent private histories (indexed by trial)
    private_histories = {}  # (group, round, agent, trial) -> vector

    # ---------- First pass: fill private_reward_vector and store history ----------
    for (group, rnd, agent), group_data in group_round_agent_groups:
        choices = group_data['choice'].astype(int).to_numpy()
        rewards = group_data['reward'].to_numpy()
        trials = group_data['trial'].astype(int).to_numpy()

        private_vector = np.zeros(n_options)
        for idx, (trial, choice, reward) in zip(group_data.index, zip(trials, choices, rewards)):
            # Store a copy of the current state (before this trial)
            df.at[idx, 'private_reward_vector'] = private_vector.copy()
            private_histories[(group, rnd, agent, trial)] = private_vector.copy()
            # Update private vector for next trial
            private_vector[choice] = reward

    # ---------- Second pass: fill social_reward_vector ----------
    for (group, rnd), group_data in group_round_groups:
        agents = group_data['agent'].unique()

        for agent in agents:
            # Trials for this agent
            agent_data = df[(df['group'] == group) & (df['round'] == rnd) & (df['agent'] == agent)]
            trials = agent_data['trial'].astype(int).to_numpy()

            for idx, trial in zip(agent_data.index, trials):
                # Gather private histories from other agents up to this trial
                other_agents = [a for a in agents if a != agent]
                social_vector = np.mean(
                    [private_histories[(group, rnd, other_agent, trial)] for other_agent in other_agents], axis=0)

                # Assign computed social vector
                df.at[idx, 'social_reward_vector'] = social_vector

    return df


def build_summary_vectors(df, n_trials=15):
    # Container for the summary
    summary_rows = []

    # Group by group and round
    group_round_groups = df.groupby(['group', 'round'], sort=False)

    for (group, rnd), group_data in group_round_groups:
        agents = group_data['agent'].unique()

        # Pre-extract data for all agents in this group/round
        agent_data = {}
        for agent in agents:
            agent_trials = group_data[group_data['agent'] == agent].sort_values('trial')
            choices = agent_trials['choice'].to_numpy()
            rewards = agent_trials['reward'].to_numpy()
            agent_data[agent] = (choices[:n_trials], rewards[:n_trials])

        # Now build a vector for each agent
        for agent in agents:
            # Own choices and rewards
            own_choices, own_rewards = agent_data[agent]

            # Collect others' choices and rewards
            others = [a for a in agents if a != agent]
            others_choices_rewards = []
            for other_agent in others:
                other_choices, other_rewards = agent_data[other_agent]
                others_choices_rewards.extend([other_choices, other_rewards])

            # Concatenate all into one vector
            summary_vector = np.concatenate(
                [[group_data['env'].values[0]]] + [own_choices, own_rewards] + others_choices_rewards
            )

            summary_rows.append({
                'group': group,
                'round': rnd,
                'agent': agent,
                'summary_vector': summary_vector
            })

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def prepare_x(simulations, n_options=121):
    x_list = []

    for idx, row in simulations.iterrows():

        # Extract private and social history vectors
        private_vector = row['private_reward_vector']
        social_vector = row['social_reward_vector']

        # Concatenate parameters and history into theta
        history_vector = np.concatenate([private_vector, social_vector])

        # One-hot encode choice
        choice = int(row['choice'])
        choice_one_hot = F.one_hot(torch.tensor(choice), num_classes=n_options).numpy()

        x_list.append(np.concatenate([choice_one_hot, history_vector]))

    x = torch.tensor(np.array(x_list), dtype=torch.float32)

    return x

def learn_likelihood(all_environments, save_dir, n_environments=5.0, n_groups_simulation=6000,
                     params=("lambda", "beta", "tau", "eps_soc"), n_options=121, n_rounds=8,
                     reward_min=-1, reward_max=1, prior_type='weakly_informed'):
    params = list(params)
    _proposal = weakly_informed_prior if prior_type == 'weakly_informed' else box_uniform_prior

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
    # simulations = extract_history_vectors(simulations)
    # plt.plot(simulations.loc[14, 'private_reward_vector']); plt.show()
    # plt.plot(simulations.loc[14, 'social_reward_vector']); plt.show()

    # cols = params + ['env']
    # theta = torch.tensor(simulations[cols].to_numpy(), dtype=torch.float32)
    # x = torch.tensor(simulations['choice'].to_numpy(), dtype=torch.float32).unsqueeze(1)
    # theta = torch.tensor(simulations[params].to_numpy(), dtype=torch.float32)
    # x = prepare_x(simulations, n_options=n_options)

    simulations_orig = simulations.sort_values(['group', 'round', 'agent', 'trial']).reset_index(drop=True)

    # save the simulations
    # simulations_orig.to_csv(save_dir / "simulations.csv", index=False)

    simulations = build_summary_vectors(simulations_orig)
    x = torch.tensor(np.array(simulations['summary_vector'].to_list()), dtype=torch.float32)
    theta = torch.tensor(simulations_orig.loc[simulations_orig.trial == 1, params].to_numpy(), dtype=torch.float32)

    trainer = SNPE(_proposal, show_progress_bars=True, density_estimator="maf")
    # trainer = NLE(_proposal, show_progress_bars=True, density_estimator="maf")
    estimator = trainer.append_simulations(theta, x)
    _inference = estimator.train()

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


    # simulations_o = extract_history_vectors(simulations_o)
    #
    # # theta_o = torch.tensor(simulations_o[params + ['env']].to_numpy(), dtype=torch.float32)
    # # x_o = torch.tensor(simulations_o['choice'].to_numpy(), dtype=torch.float32).unsqueeze(1).T
    # _x_o = prepare_x(simulations_o)
    # _theta_o = torch.tensor(simulations_o[params].to_numpy(), dtype=torch.float32)


    simulations_o_orig = simulations_o.sort_values(['group', 'round', 'agent', 'trial']).reset_index(drop=True)
    simulations_o = build_summary_vectors(simulations_o_orig)
    _x_o = torch.tensor(np.array(simulations_o['summary_vector'].to_list()), dtype=torch.float32)
    _theta_o = torch.tensor(simulations_o_orig.loc[simulations_o_orig.trial == 1, params].to_numpy(), dtype=torch.float32)

    return _theta_o, _theta_o_df, _x_o, simulations_o.agent.values


def human_observations(subj_data_all, group_id=None):
    if group_id is not None:
        subj_data = subj_data_all[(subj_data_all['group'] == group_id)].copy()
    else:
        subj_data = subj_data_all.copy()

    # subj_data = extract_history_vectors(subj_data)
    # _x_o = prepare_x(subj_data)

    subj_data = subj_data.sort_values(['group', 'round', 'agent', 'trial']).reset_index(drop=True)
    subj_data = build_summary_vectors(subj_data)
    _x_o = torch.tensor(np.array(subj_data['summary_vector'].to_list()), dtype=torch.float32)

    return _x_o, subj_data.agent.values


def save_posterior(save_dir, prior_type, density_estimator):
    prior  = weakly_informed_prior if prior_type == 'weakly_informed' else box_uniform_prior
    trainer = SNPE(prior, density_estimator="maf")
    posterior = trainer.build_posterior(
        density_estimator=density_estimator,
        prior=prior,
    )

    with open(save_dir / "posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)


def fit_posterior(density_estimator, _theta_o_df, _x_o, _save_dir, num_samples=50_000,
                  params=("lambda", "beta", "tau", "eps_soc"), prior_type='weakly_informed'):
    params = list(params)
    mcmc_parameters = dict(
        num_chains=7,
        thin=1,
        warmup_steps=500,
        init_strategy="proposal",
        num_workers=8
    )
    prior  = weakly_informed_prior if prior_type == 'weakly_informed' else box_uniform_prior
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

    trainer = SNPE(prior, density_estimator="maf")
    # trainer = NLE(prior, density_estimator="maf")
    posterior = trainer.build_posterior(
        density_estimator=density_estimator,
        prior=prior,
        # mcmc_method="nuts_pyro",  # "nuts_pyro" "slice_np_vectorized"
        # mcmc_parameters=mcmc_parameters,
    )

    with open(save_dir / "posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)

    posterior_sample = posterior.sample_batched((num_samples,), x=_x_o.T)
    # posterior_sample shape is (num_x, num_samples, num_theta)
    # need to reshape to (num_x * num_samples, num_theta)
    posterior_sample = posterior_sample.reshape(-1, len(params))


    # posterior_sample = posterior.sample((num_samples,), x=_x_o.T)

    # posterior = DirectPosterior(density_estimator, prior=prior)
    # posterior_sample = posterior.sample_batched((num_samples,), x=x_o.T)

    # inference_data = posterior.get_arviz_inference_data()
    # with az.style.context("arviz-darkgrid"):
    #     az.plot_trace(inference_data, compact=False)
    #
    # plt.savefig(_save_dir / "traceplot.png", dpi=300)
    # plt.close()


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

    return posterior_sample


if __name__ == "__main__":
    # parce arguments
    import argparse
    parser = argparse.ArgumentParser(description='Parameter recovery')
    # n_groups_simulation, prior_type, num_samples
    parser.add_argument('--n_groups_simulation', type=int, default=10_000)
    parser.add_argument('--prior_type', type=str, default='weakly_informed')
    parser.add_argument('--num_samples', type=int, default=50_000)
    args = parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(42)

    save_dir = Path("parameter_recovery") / datetime.now().strftime("%Y%m%d-%H%M%S-with-env")
    save_dir.mkdir(parents=True, exist_ok=True)

    path = 'data/environments'
    json_files = [file for file in os.listdir(path) if file.endswith('_canon.json')]
    all_environments = []
    for file in json_files:
        f=open(os.path.join(path, file))
        all_environments.append(json.load(f))


    inference = learn_likelihood(all_environments, save_dir, n_environments=len(all_environments[0]),
                                 n_groups_simulation=args.n_groups_simulation, n_rounds=8, prior_type=args.prior_type)

    save_posterior(save_dir, args.prior_type, inference)

    ground_truths = [
        (1.11, 0.33, 0.03, 12.55),
        (0.88, 0.22, 0.01, 15.33),
        (1.44, 0.44, 0.02, 9.77),
        (0.77, 0.55, 0.05, 10.22),
        (1.11, 0.44, 0.02, 0.77),
        (1.11, 0.33, 0.1, 12.55),
        (0.11, 0.33, 0.03, 12.55),
    ]

    for i, gt in enumerate(ground_truths):
        _, theta_o_df, x_o, agent = simulate_observations(all_environments, n_environments=1, n_groups_simulation=1,
                                                          ground_truth_params=gt, simulated_agents=(0,))
        save_dir2 = save_dir / f'param_recovery_{i}'
        save_dir2.mkdir(parents=True, exist_ok=True)
        posterior_samples = fit_posterior(inference, theta_o_df, x_o[agent == 0].T, save_dir2,
                                          num_samples=args.num_samples, prior_type=args.prior_type)


    subj_data_all = pd.read_csv("./data/e1_data.csv")
    fit_results = pd.read_csv("./data/e1_fitting_data/fit+pars_e1.csv")
    for group_id in fit_results.group.unique():
        for agent_id in [1, 2, 3, 4]:
            h_x_o, h_agent = human_observations(subj_data_all, group_id=group_id)
            h_theta_o_df = pd.DataFrame(columns=("lambda", "beta", "tau", "eps_soc"), index=[0], dtype=float)
            mask = (fit_results.group == group_id) & (fit_results.model == 'SG') & (fit_results.agent == agent_id)
            h_theta_o_df.loc[0,:] = fit_results.loc[mask, ["lambda", "beta", "tau", "par"]].values

            save_dir2 = save_dir / f'g_{group_id}_a_{agent_id}'
            save_dir2.mkdir(parents=True, exist_ok=True)
            posterior_samples = fit_posterior(inference, h_theta_o_df, h_x_o[h_agent == agent_id].T, save_dir2,
                                              num_samples=args.num_samples, prior_type=args.prior_type)

            # average
            fit_results.loc[mask, ["lambda_sbi", "beta_sbi", "tau_sbi", "eps_soc_sbi"]] = posterior_samples.mean(axis=0)
            fit_results.to_csv(save_dir / "fit+pars_e1_sbi.csv", index=False)