"""
data_plots.py — Generate paper-ready figures from madqn and Dqn results.

Usage:
  python data_plots.py --results-dir results/
  python data_plots.py --results-dir results/ --save-dir figures/
"""


import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  True,
    "axes.spines.right":True,
    "axes.grid":        True,
    "grid.alpha":       0.3,
})

COLORS = {
    "E-MATRO++":       "C0",
    "IDQN":          "C1",
    "C-DQN":           "C2",
    "Random":        "C3",
    "GroupAll":      "C4",
    "SplitAll":      "C5",
}



def smooth(values, window=50):
    kernel = np.ones(window) / window
    pad    = np.pad(values, (window//2, window//2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")[:len(values)]



def fig1_reward_curve(df_e_matro, df_dqn, df_idqn,  save_dir): 
    fig, ax = plt.subplots(figsize=(7, 3.5))

    ep  = df_e_matro["episode"].values
    raw = df_e_matro["avg_reward"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2 = df_dqn["episode"].values
    raw2 = df_dqn["avg_reward"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3 = df_idqn["episode"].values
    raw3 = df_idqn["avg_reward"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")

    
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Average reward", fontsize=11)
    
    ax.grid(True, alpha=0.3)
    

    ax.legend()
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize='x-small',
    #  columnspacing=0.8, handlelength=1.2)
    # fig.subplots_adjust(top=0.85)
    fig.tight_layout()
    path = os.path.join(save_dir, "reward_curve.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_cdf_reward(df_e_matro, df_dqn, df_idqn,  save_dir): 
    fig, ax = plt.subplots(figsize=(7, 3.5))


    rewards_ma = np.sort(df_e_matro["avg_reward"].values)
    cdf_ma     = np.arange(1, len(rewards_ma) + 1) / len(rewards_ma)
    ax.plot(rewards_ma, cdf_ma, color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    rewards_dqn = np.sort(df_dqn["avg_reward"].values)
    cdf_dqn     = np.arange(1, len(rewards_dqn) + 1) / len(rewards_dqn)
    ax.plot(rewards_dqn, cdf_dqn, color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    rewards_idqn = np.sort(df_idqn["avg_reward"].values)
    cdf_idqn     = np.arange(1, len(rewards_idqn) + 1) / len(rewards_idqn)
    ax.plot(rewards_idqn, cdf_idqn, color=COLORS["IDQN"], linewidth=1.8, label="IDQN")


    ax.set_xlabel("Average reward", fontsize=11)
    ax.set_ylabel("Empirical CDF", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "cdf_reward.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")   


def fig_ccdf_reward(df_e_matro, df_dqn, df_idqn, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    rewards_ma = np.sort(df_e_matro["avg_reward"].values)
    ccdf_ma    = 1 - np.arange(1, len(rewards_ma) + 1) / len(rewards_ma)
    ax.plot(rewards_ma, ccdf_ma, color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    rewards_dqn = np.sort(df_dqn["avg_reward"].values)
    ccdf_dqn    = 1 - np.arange(1, len(rewards_dqn) + 1) / len(rewards_dqn)
    ax.plot(rewards_dqn, ccdf_dqn, color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    rewards_idqn = np.sort(df_idqn["avg_reward"].values)
    ccdf_idqn    = 1 - np.arange(1, len(rewards_idqn) + 1) / len(rewards_idqn)
    ax.plot(rewards_idqn, ccdf_idqn, color=COLORS["IDQN"], linewidth=1.8, label="IDQN")

    ax.set_xlabel("Average reward", fontsize=11)
    ax.set_ylabel("Complementary cdf", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "ccdf_reward.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig2_energy_curve(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    ep  = df_e_matro["episode"].values
    raw = df_e_matro["avg_energy"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["avg_energy"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3 = df_idqn["episode"].values
    raw3 = df_idqn["avg_energy"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")


    ep4 = df_random["episode"].values
    raw4 = df_random["avg_energy"].values
    ax.plot(ep4, raw4, alpha=0.2, color=COLORS["Random"], linewidth=0.6)
    ax.plot(ep4, smooth(raw4), color=COLORS["Random"], linewidth=1.8, label="Random")

    ep5 = df_group_all["episode"].values
    raw5 = df_group_all["avg_energy"].values
    ax.plot(ep5, raw5, alpha=0.2, color=COLORS["GroupAll"], linewidth=0.6)
    ax.plot(ep5, smooth(raw5), color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    ep6 = df_split_all["episode"].values
    raw6 = df_split_all["avg_energy"].values
    ax.plot(ep6, raw6, alpha=0.2, color=COLORS["SplitAll"], linewidth=0.6)
    ax.plot(ep6, smooth(raw6), color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode energy (kWh)", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "energy_curve.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig3_migrations_curve(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["total_migrations"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["total_migrations"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["total_migrations"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")

    ep4 = df_random["episode"].values
    raw4 = df_random["total_migrations"].values
    ax.plot(ep4, raw4, alpha=0.2, color=COLORS["Random"], linewidth=0.6)
    ax.plot(ep4, smooth(raw4), color=COLORS["Random"], linewidth=1.8, label="Random")

    ep5 = df_group_all["episode"].values
    raw5 = df_group_all["total_migrations"].values
    ax.plot(ep5, raw5, alpha=0.2, color=COLORS["GroupAll"], linewidth=0.6)
    ax.plot(ep5, smooth(raw5), color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    ep6 = df_split_all["episode"].values
    raw6 = df_split_all["total_migrations"].values
    ax.plot(ep6, raw6, alpha=0.2, color=COLORS["SplitAll"], linewidth=0.6)
    ax.plot(ep6, smooth(raw6), color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Number of migrations", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "migrations_curve.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")



def fig4_loss_curve(df_e_matro, df_dqn, df_idqn, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["avg_loss"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["avg_loss"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["avg_loss"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Training episode loss", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "loss_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig5_conflicts_curve(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["total_conflicts"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["total_conflicts"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["total_conflicts"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")


    ep4 = df_random["episode"].values
    raw4 = df_random["total_conflicts"].values
    ax.plot(ep4, raw4, alpha=0.2, color=COLORS["Random"], linewidth=0.6)
    ax.plot(ep4, smooth(raw4), color=COLORS["Random"], linewidth=1.8, label="Random")

    ep5 = df_group_all["episode"].values
    raw5 = df_group_all["total_conflicts"].values
    ax.plot(ep5, raw5, alpha=0.2, color=COLORS["GroupAll"], linewidth=0.6)
    ax.plot(ep5, smooth(raw5), color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    ep6 = df_split_all["episode"].values
    raw6 = df_split_all["total_conflicts"].values
    ax.plot(ep6, raw6, alpha=0.2, color=COLORS["SplitAll"], linewidth=0.6)
    ax.plot(ep6, smooth(raw6), color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")

    

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Number of conflicts", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "conflicts_curve.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig6_nodes_curve(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["avg_nodes_active"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["avg_nodes_active"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["avg_nodes_active"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")

    ep4 = df_random["episode"].values
    raw4 = df_random["avg_nodes_active"].values
    ax.plot(ep4, raw4, alpha=0.2, color=COLORS["Random"], linewidth=0.6)
    ax.plot(ep4, smooth(raw4), color=COLORS["Random"], linewidth=1.8, label="Random")

    ep5 = df_group_all["episode"].values
    raw5 = df_group_all["avg_nodes_active"].values
    ax.plot(ep5, raw5, alpha=0.2, color=COLORS["GroupAll"], linewidth=0.6)
    ax.plot(ep5, smooth(raw5), color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    ep6 = df_split_all["episode"].values
    raw6 = df_split_all["avg_nodes_active"].values
    ax.plot(ep6, raw6, alpha=0.2, color=COLORS["SplitAll"], linewidth=0.6)
    ax.plot(ep6, smooth(raw6), color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")


    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Number of active nodes", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "nodes_curve.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig8_energy_saving_curve(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["energy_saving_%"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["energy_saving_%"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["energy_saving_%"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")

    ep4 = df_random["episode"].values
    raw4 = df_random["energy_saving_%"].values
    ax.plot(ep4, raw4, alpha=0.2, color=COLORS["Random"], linewidth=0.6)
    ax.plot(ep4, smooth(raw4), color=COLORS["Random"], linewidth=1.8, label="Random")

    ep5 = df_group_all["episode"].values
    raw5 = df_group_all["energy_saving_%"].values
    ax.plot(ep5, raw5, alpha=0.2, color=COLORS["GroupAll"], linewidth=0.6)
    ax.plot(ep5, smooth(raw5), color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    ep6 = df_split_all["episode"].values
    raw6 = df_split_all["energy_saving_%"].values
    ax.plot(ep6, raw6, alpha=0.2, color=COLORS["SplitAll"], linewidth=0.6)
    ax.plot(ep6, smooth(raw6), color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")


    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Energy saving (%)", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "energy_saving_curve.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig9_inference_curve(df_e_matro, df_dqn, df_idqn, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["avg_inference_ms"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["avg_inference_ms"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["avg_inference_ms"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")


    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Average inference time (ms)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "inference_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")







def fig10_training_curve(df_e_matro, df_dqn, df_idqn, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["training_time"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["training_time"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["training_time"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Training time (s)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "training_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig11_peak_load_curve(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["peak_load"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["peak_load"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["peak_load"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")


    ep4 = df_random["episode"].values
    raw4 = df_random["peak_load"].values
    ax.plot(ep4, raw4, alpha=0.2, color=COLORS["Random"], linewidth=0.6)
    ax.plot(ep4, smooth(raw4), color=COLORS["Random"], linewidth=1.8, label="Random")

    ep5 = df_group_all["episode"].values
    raw5 = df_group_all["peak_load"].values
    ax.plot(ep5, raw5, alpha=0.2, color=COLORS["GroupAll"], linewidth=0.6)
    ax.plot(ep5, smooth(raw5), color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    ep6 = df_split_all["episode"].values
    raw6 = df_split_all["peak_load"].values
    ax.plot(ep6, raw6, alpha=0.2, color=COLORS["SplitAll"], linewidth=0.6)
    ax.plot(ep6, smooth(raw6), color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")


    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Average peak load", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "peak_load_curve.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig12_energy_efficiency_curve(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["avg_energy_eff"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["avg_energy_eff"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["avg_energy_eff"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")

    ep4  = df_random["episode"].values
    raw4 = df_random["energy_efficiency"].values
    ax.plot(ep4, raw4, alpha=0.2, color=COLORS["Random"], linewidth=0.6)
    ax.plot(ep4, smooth(raw4), color=COLORS["Random"], linewidth=1.8, label="Random")

    ep5  = df_group_all["episode"].values
    raw5 = df_group_all["energy_efficiency"].values
    ax.plot(ep5, raw5, alpha=0.2, color=COLORS["GroupAll"], linewidth=0.6)
    ax.plot(ep5, smooth(raw5), color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    ep6  = df_split_all["episode"].values
    raw6 = df_split_all["energy_efficiency"].values
    ax.plot(ep6, raw6, alpha=0.2, color=COLORS["SplitAll"], linewidth=0.6)
    ax.plot(ep6, smooth(raw6), color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Average energy efficiency", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "energy_efficiency_curve.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig13_ocloud_util_curve(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["ocloud_utilization"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["ocloud_utilization"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["ocloud_utilization"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")

    ep4  = df_random["episode"].values
    raw4 = df_random["ocloud_utilization"].values
    ax.plot(ep4, raw4, alpha=0.2, color=COLORS["Random"], linewidth=0.6)
    ax.plot(ep4, smooth(raw4), color=COLORS["Random"], linewidth=1.8, label="Random")

    ep5  = df_group_all["episode"].values
    raw5 = df_group_all["ocloud_utilization"].values
    ax.plot(ep5, raw5, alpha=0.2, color=COLORS["GroupAll"], linewidth=0.6)
    ax.plot(ep5, smooth(raw5), color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    ep6  = df_split_all["episode"].values
    raw6 = df_split_all["ocloud_utilization"].values
    ax.plot(ep6, raw6, alpha=0.2, color=COLORS["SplitAll"], linewidth=0.6)
    ax.plot(ep6, smooth(raw6), color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Average O-Cloud utilization", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "ocloud_util_curve.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig14_capacity_violations_curve(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["capacity_violations"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["capacity_violations"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["capacity_violations"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")

    ep4  = df_random["episode"].values
    raw4 = df_random["capacity_violations"].values
    ax.plot(ep4, raw4, alpha=0.2, color=COLORS["Random"], linewidth=0.6)
    ax.plot(ep4, smooth(raw4), color=COLORS["Random"], linewidth=1.8, label="Random")

    ep5  = df_group_all["episode"].values
    raw5 = df_group_all["capacity_violations"].values
    ax.plot(ep5, raw5, alpha=0.2, color=COLORS["GroupAll"], linewidth=0.6)
    ax.plot(ep5, smooth(raw5), color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    ep6  = df_split_all["episode"].values
    raw6 = df_split_all["capacity_violations"].values
    ax.plot(ep6, raw6, alpha=0.2, color=COLORS["SplitAll"], linewidth=0.6)
    ax.plot(ep6, smooth(raw6), color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Capacity violations", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "capacity_violations_curve.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")



def fig15_conflict_rate_curve(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["conflict_rate"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["conflict_rate"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["conflict_rate"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")

    ep4  = df_random["episode"].values
    raw4 = df_random["conflict_rate"].values
    ax.plot(ep4, raw4, alpha=0.2, color=COLORS["Random"], linewidth=0.6)
    ax.plot(ep4, smooth(raw4), color=COLORS["Random"], linewidth=1.8, label="Random")

    ep5  = df_group_all["episode"].values
    raw5 = df_group_all["conflict_rate"].values
    ax.plot(ep5, raw5, alpha=0.2, color=COLORS["GroupAll"], linewidth=0.6)
    ax.plot(ep5, smooth(raw5), color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    ep6  = df_split_all["episode"].values
    raw6 = df_split_all["conflict_rate"].values
    ax.plot(ep6, raw6, alpha=0.2, color=COLORS["SplitAll"], linewidth=0.6)
    ax.plot(ep6, smooth(raw6), color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Conflict rate", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "conflict_rate_curve.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig16_migration_success_rate_curve(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["migration_success_rate"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["migration_success_rate"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["migration_success_rate"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")

    ep4  = df_random["episode"].values
    raw4 = df_random["migration_success_rate"].values
    ax.plot(ep4, raw4, alpha=0.2, color=COLORS["Random"], linewidth=0.6)
    ax.plot(ep4, smooth(raw4), color=COLORS["Random"], linewidth=1.8, label="Random")

    ep5  = df_group_all["episode"].values
    raw5 = df_group_all["migration_success_rate"].values
    ax.plot(ep5, raw5, alpha=0.2, color=COLORS["GroupAll"], linewidth=0.6)
    ax.plot(ep5, smooth(raw5), color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    ep6  = df_split_all["episode"].values
    raw6 = df_split_all["migration_success_rate"].values
    ax.plot(ep6, raw6, alpha=0.2, color=COLORS["SplitAll"], linewidth=0.6)
    ax.plot(ep6, smooth(raw6), color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Migration success rate", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "migration_success_rate_curve.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")




def fig17_OCloud_bar_chart(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Converged window — last 20% of episodes
    def conv(df):
        n = max(1, int(len(df) * 0.2))
        return df.tail(n)["ocloud_utilization"].dropna().astype(float).values

    x1 = conv(df_e_matro)
    x2 = conv(df_dqn)
    x3 = conv(df_idqn)
    x4 = conv(df_random)
    x5 = conv(df_group_all)
    x6 = conv(df_split_all)

    # Bins matching your three demand scenarios
    # bins   = [0.0, 0.30, 0.55, 1.01]
    # labels = ['Light\n(0–0.30)', 'Medium\n(0.30–0.55)', 'High\n(0.55–1.0)']

    bins   = [0.0, 0.40, 0.60, 1.0]
    labels = ['Light\n(0–0.40)', 'Medium\n(0.40–0.60)', 'High\n(0.60–1.0)']

    def categorize(data, bins):
        hist, _ = np.histogram(data, bins=bins)
        return (hist / len(data)) * 100 if len(data) > 0 else np.zeros(len(bins)-1)

    x1_pct = categorize(x1, bins)
    x2_pct = categorize(x2, bins)
    x3_pct = categorize(x3, bins)

    x4_pct = categorize(x4, bins)
    x5_pct = categorize(x5, bins)
    x6_pct = categorize(x6, bins)

    n_bars = 6
    group_width = 0.7          # total width used by all 6 bars for one category (out of 1.0 available)
    bar_width = group_width / n_bars   # = 0.1167


    #bar_width = 0.25
    # r1 = np.arange(len(labels))
    # r2 = r1 + bar_width
    # r3 = r2 + bar_width

    # r4 = r3 + bar_width
    # r5 = r4 + bar_width
    # r6 = r5 + bar_width

    # ax.bar(r1, x1_pct, width=bar_width, label='E-MATRO', color=COLORS["E-MATRO"])
    # ax.bar(r2, x2_pct, width=bar_width, label='C-DQN',   color=COLORS["C-DQN"])
    # ax.bar(r3, x3_pct, width=bar_width, label='IDQN',    color=COLORS["IDQN"])


    # ax.bar(r4, x4_pct, width=bar_width, label='Random',  color=COLORS["Random"])
    # ax.bar(r5, x5_pct, width=bar_width, label='GroupAll',color=COLORS["GroupAll"])
    # ax.bar(r6, x6_pct, width=bar_width, label='SplitAll',color=COLORS["SplitAll"])


    x = np.arange(len(labels))
    data = [x1_pct, x2_pct, x3_pct, x4_pct, x5_pct, x6_pct]
    lbls  = ['E-MATRO++', 'C-DQN', 'IDQN', 'Random', 'GroupAll', 'SplitAll']
    cols  = [COLORS[l] for l in lbls]

    for i, (d, l, c) in enumerate(zip(data, lbls, cols)):
        offset = (i - (n_bars - 1) / 2) * bar_width   # centers the whole group on x
        ax.bar(x + offset, d, width=bar_width, label=l, color=c)


    ax.set_xlabel("Regime of O-Cloud utilization", fontsize=11)
    ax.set_ylabel("Percentage of utilization (%)", fontsize=11)
    ax.set_xticks(x)
    #ax.set_xticks(r1 + 2.5 * bar_width)
    ax.set_xticklabels(labels, fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "ocloud_utilization_bar_chart.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_cdf_CloudUtil(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))


    util_dqn = np.sort(df_dqn["ocloud_utilization"].values)
    cdf_dqn     = np.arange(1, len(util_dqn) + 1) / len(util_dqn)
    ax.plot(util_dqn, cdf_dqn, color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    util_ematro = np.sort(df_e_matro["ocloud_utilization"].values)
    cdf_ma     = np.arange(1, len(util_ematro) + 1) / len(util_ematro)
    ax.plot(util_ematro, cdf_ma, color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    
    util_idqn = np.sort(df_idqn["ocloud_utilization"].values)
    cdf_idqn     = np.arange(1, len(util_idqn) + 1) / len(util_idqn)
    ax.plot(util_idqn, cdf_idqn, color=COLORS["IDQN"], linewidth=1.8, label="IDQN")


    util_random = np.sort(df_random["ocloud_utilization"].values)
    cdf_random     = np.arange(1, len(util_random) + 1) / len(util_random)
    ax.plot(util_random, cdf_random, color=COLORS["Random"], linewidth=1.8, label="Random") 

    util_group_all = np.sort(df_group_all["ocloud_utilization"].values)
    cdf_group_all     = np.arange(1, len(util_group_all) + 1) / len(util_group_all)
    ax.plot(util_group_all, cdf_group_all, color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    util_split_all = np.sort(df_split_all["ocloud_utilization"].values)
    cdf_split_all     = np.arange(1, len(util_split_all) + 1) / len(util_split_all)
    ax.plot(util_split_all, cdf_split_all, color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")


    ax.set_xlabel("O-Cloud utilization", fontsize=11)
    ax.set_ylabel("Empirical CDF", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "cdf_ocloud_utilization.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")  



def fig_cdf_migration(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    
    mig_ematro = np.sort(df_e_matro["total_migrations"].values)
    cdf_ma     = np.arange(1, len(mig_ematro) + 1) / len(mig_ematro)
    ax.plot(mig_ematro, cdf_ma, color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    mig_dqn = np.sort(df_dqn["total_migrations"].values)
    cdf_dqn     = np.arange(1, len(mig_dqn) + 1) / len(mig_dqn)
    ax.plot(mig_dqn, cdf_dqn, color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    mig_idqn = np.sort(df_idqn["total_migrations"].values)
    cdf_idqn     = np.arange(1, len(mig_idqn) + 1) / len(mig_idqn)
    ax.plot(mig_idqn, cdf_idqn, color=COLORS["IDQN"], linewidth=1.8, label="IDQN")

    mig_random = np.sort(df_random["total_migrations"].values)
    cdf_random     = np.arange(1, len(mig_random) + 1) / len(mig_random)
    ax.plot(mig_random, cdf_random, color=COLORS["Random"], linewidth=1.8, label="Random")

    mig_group_all = np.sort(df_group_all["total_migrations"].values)
    cdf_group_all     = np.arange(1, len(mig_group_all) + 1) / len(mig_group_all)
    ax.plot(mig_group_all, cdf_group_all, color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")    

    mig_split_all = np.sort(df_split_all["total_migrations"].values)
    cdf_split_all     = np.arange(1, len(mig_split_all) + 1) / len(mig_split_all)
    ax.plot(mig_split_all, cdf_split_all, color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")


    ax.set_xlabel("Total migrations", fontsize=11)
    ax.set_ylabel("Empirical CDF", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "cdf_total_migrations.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}") 


def fig_cdf_capacity_violations(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))


    cv_ematro = np.sort(df_e_matro["capacity_violations"].values)
    cdf_ma     = np.arange(1, len(cv_ematro) + 1) / len(cv_ematro)
    ax.plot(cv_ematro, cdf_ma, color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    cv_dqn = np.sort(df_dqn["capacity_violations"].values)
    cdf_dqn     = np.arange(1, len(cv_dqn) + 1) / len(cv_dqn)
    ax.plot(cv_dqn, cdf_dqn, color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    cv_idqn = np.sort(df_idqn["capacity_violations"].values)
    cdf_idqn     = np.arange(1, len(cv_idqn) + 1) / len(cv_idqn)
    ax.plot(cv_idqn, cdf_idqn, color=COLORS["IDQN"], linewidth=1.8, label="IDQN")


    cv_random = np.sort(df_random["capacity_violations"].values)
    cdf_random     = np.arange(1, len(cv_random) + 1) / len(cv_random)
    ax.plot(cv_random, cdf_random, color=COLORS["Random"], linewidth=1.8, label="Random") 

    cv_group_all = np.sort(df_group_all["capacity_violations"].values)
    cdf_group_all     = np.arange(1, len(cv_group_all) + 1) / len(cv_group_all)
    ax.plot(cv_group_all, cdf_group_all, color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    cv_split_all = np.sort(df_split_all["capacity_violations"].values)
    cdf_split_all     = np.arange(1, len(cv_split_all) + 1) / len(cv_split_all)
    ax.plot(cv_split_all, cdf_split_all, color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")


    ax.set_xlabel("Capacity violations", fontsize=11)
    ax.set_ylabel("Empirical CDF", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "cdf_capacity_violations.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_cdf_conflict_rate(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))


    cr_ematro = np.sort(df_e_matro["conflict_rate"].values)
    cdf_ma     = np.arange(1, len(cr_ematro) + 1) / len(cr_ematro)
    ax.plot(cr_ematro, cdf_ma, color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    cr_dqn = np.sort(df_dqn["conflict_rate"].values)
    cdf_dqn     = np.arange(1, len(cr_dqn) + 1) / len(cr_dqn)
    ax.plot(cr_dqn, cdf_dqn, color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    cr_idqn = np.sort(df_idqn["conflict_rate"].values)
    cdf_idqn     = np.arange(1, len(cr_idqn) + 1) / len(cr_idqn)
    ax.plot(cr_idqn, cdf_idqn, color=COLORS["IDQN"], linewidth=1.8, label="IDQN")


    cr_random = np.sort(df_random["conflict_rate"].values)
    cdf_random     = np.arange(1, len(cr_random) + 1) / len(cr_random)
    ax.plot(cr_random, cdf_random, color=COLORS["Random"], linewidth=1.8, label="Random") 

    cr_group_all = np.sort(df_group_all["conflict_rate"].values)
    cdf_group_all     = np.arange(1, len(cr_group_all) + 1) / len(cr_group_all)
    ax.plot(cr_group_all, cdf_group_all, color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    cr_split_all = np.sort(df_split_all["conflict_rate"].values)
    cdf_split_all     = np.arange(1, len(cr_split_all) + 1) / len(cr_split_all)
    ax.plot(cr_split_all, cdf_split_all, color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")


    ax.set_xlabel("Conflict rate", fontsize=11)
    ax.set_ylabel("Empirical CDF", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "cdf_conflict_rate.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_bar_chart_migration(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Converged window — last 20% of episodes
    def conv(df):
        n = max(1, int(len(df) * 0.2))
        return df.tail(n)["total_migrations"].dropna().astype(float).values

    x1 = conv(df_e_matro)
    x2 = conv(df_dqn)
    x3 = conv(df_idqn)

    x4 = conv(df_random)
    x5 = conv(df_group_all)
    x6 = conv(df_split_all)


    #bins   = [0.0, 30, 42, 100]#12rus_9nodes
    #labels = ['Light\n(0–30)', 'Medium\n(30–42)', 'High\n(42–100)']

    
    bins   = [0.0, 18, 22, 27]#8rus_6nodes
    labels = ['Light\n(0–18)', 'Medium\n(18–22)', 'High\n(22–27)']


    #bins   = [0.0, 6, 9, 17]#4rus_4nodes
    #labels = ['Light\n(0–6)', 'Medium\n(6–9)', 'High\n(9–17)']



    def categorize(data, bins):
        hist, _ = np.histogram(data, bins=bins)
        return (hist / len(data)) * 100 if len(data) > 0 else np.zeros(len(bins)-1)

    x1_pct = categorize(x1, bins)
    x2_pct = categorize(x2, bins)
    x3_pct = categorize(x3, bins)

    x4_pct = categorize(x4, bins)
    x5_pct = categorize(x5, bins)
    x6_pct = categorize(x6, bins)

    n_bars = 6
    group_width = 0.7          # total width used by all 6 bars for one category (out of 1.0 available)
    bar_width = group_width / n_bars   # = 0.1167

    # r1 = np.arange(len(labels))
    # r2 = r1 + bar_width
    # r3 = r2 + bar_width

    # r4 = r3 + bar_width
    # r5 = r4 + bar_width
    # r6 = r5 + bar_width


    # ax.bar(r1, x1_pct, width=bar_width, label='IDQN', color=COLORS["IDQN"])
    # ax.bar(r2, x2_pct, width=bar_width, label='C-DQN',   color=COLORS["C-DQN"])
    # ax.bar(r3, x3_pct, width=bar_width, label='E-MATRO', color=COLORS["E-MATRO"])

    # ax.bar(r4, x4_pct, width=bar_width, label='Random',  color=COLORS["Random"])
    # ax.bar(r5, x5_pct, width=bar_width, label='GroupAll',color=COLORS["GroupAll"])
    # ax.bar(r6, x6_pct, width=bar_width, label='SplitAll',color=COLORS["SplitAll"])


    x = np.arange(len(labels))
    data = [x1_pct, x2_pct, x3_pct, x4_pct, x5_pct, x6_pct]
    

    lbls  = ['E-MATRO++', 'C-DQN', 'IDQN', 'Random', 'GroupAll', 'SplitAll']
    cols  = [COLORS[l] for l in lbls]

    for i, (d, l, c) in enumerate(zip(data, lbls, cols)):
        offset = (i - (n_bars - 1) / 2) * bar_width   # centers the whole group on x
        ax.bar(x + offset, d, width=bar_width, label=l, color=c)


    ax.set_xlabel("Regime of workload migrations", fontsize=11)
    ax.set_ylabel("Percentage of migrations (%)", fontsize=11)
    #ax.set_xticks(r2)
    #ax.set_xticks(r1 + 2.5 * bar_width)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "workload_migrations_bar_chart.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_cdf_inference_time(df_e_matro, df_dqn, df_idqn, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))  

    
    
    
    inf_ematro = np.sort(df_e_matro["avg_inference_ms"].values)
    cdf_ma     = np.arange(1, len(inf_ematro) + 1) / len(inf_ematro)
    ax.plot(inf_ematro, cdf_ma, color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    inf_dqn = np.sort(df_dqn["avg_inference_ms"].values)
    cdf_dqn     = np.arange(1, len(inf_dqn) + 1) / len(inf_dqn)
    ax.plot(inf_dqn, cdf_dqn, color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    inf_idqn = np.sort(df_idqn["avg_inference_ms"].values)
    cdf_idqn     = np.arange(1, len(inf_idqn) + 1) / len(inf_idqn)
    ax.plot(inf_idqn, cdf_idqn, color=COLORS["IDQN"], linewidth=1.8, label="IDQN")


    ax.set_xlabel("Inference time (ms)", fontsize=11)
    ax.set_ylabel("Empirical CDF", fontsize=11)
    ax.legend()
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize='x-small',
    #  columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "cdf_inference_time.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")




def fig_energy_eff_learning_rate(save_dir):

    learning_rate = [1e-4, 5*1e-4, 1e-3, 5*1e-3]

    ematro_energyconsumption = [ 802.5, 637.5, 637.5, 555.0]

    idrl_energyconsumption = [ 1050.0, 885.0, 1050.0, 885.0]

    cdrl_energyconsumption = [ 967.5, 885.0, 1050.0, 1050.0]


    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(learning_rate, ematro_energyconsumption, marker='o', color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")
    ax.plot(learning_rate, idrl_energyconsumption, marker='o', color=COLORS["IDQN"], linewidth=1.8, label="IDQN")
    ax.plot(learning_rate, cdrl_energyconsumption, marker='o', color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ax.set_xlabel("Learning rate", fontsize=11)
    ax.set_ylabel("Energy consumption (kWh)", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize='x-small',
    columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "energy_eff_learning_rate.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")



def fig_reward_learning_rate(save_dir):
    learning_rate = [1e-4, 5*1e-4, 1e-3, 5*1e-3]

    ematro_reward = [ 0.1167, 0.1917, 0.1917, 0.2194]
    idrl_reward = [ 0.1500, 0.2000, 0.1583, 0.1583]
    cdrl_reward = [ 0.1785, 0.1500, 0.1612, 0.1583]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(learning_rate, ematro_reward, marker='o', color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")
    ax.plot(learning_rate, idrl_reward, marker='o', color=COLORS["IDQN"], linewidth=1.8, label="IDQN")
    ax.plot(learning_rate, cdrl_reward, marker='o', color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ax.set_xlabel("Learning rate", fontsize=11)
    ax.set_ylabel("Average reward", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "reward_learning_rate.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")



def fig_migration_learning_rate(save_dir):

    learning_rate = [1e-4, 5*1e-4, 1e-3, 5*1e-3]
    ematro_migrations = [21, 19, 19, 18]
    idrl_migrations = [ 25, 23, 27,  22]
    cdrl_migrations = [ 24, 27, 23, 26]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(learning_rate, ematro_migrations, marker='o', color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")
    ax.plot(learning_rate, idrl_migrations, marker='o', color=COLORS["IDQN"], linewidth=1.8, label="IDQN")
    ax.plot(learning_rate, cdrl_migrations, marker='o', color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ax.set_xlabel("Learning rate", fontsize=11)
    ax.set_ylabel("Migrations", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "migration_learning_rate.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")



def fig_sensitivity_analysis_learning_rate(save_dir):

    learning_rate = [1e-4, 5*1e-4, 1e-3, 0.002, 0.003, 0.004 ,5*1e-3]

    #ematro_energyconsumption = [ 802.5, 637.5, 637.5,  885.0, 720.0, 885.0 , 555.0]
    ematro_energyconsumption = [ 885.0,  885.0, 802.5, 637.5, 637.5, 555.0, 720.0]
    #idrl_energyconsumption = [ 1050.0, 885.0, 1050.0, 885.0, 967.5 , 1050.0 ,885.0]
    idrl_energyconsumption = [ 1050.0, 1050.0, 1050.0 , 967.5, 885.0,  885.0,  885.0]
    cdrl_energyconsumption = [ 1050.0, 1050.0, 1050.0, 1050.0, 967.5, 885.0, 885.0 ]

    #ematro_reward = [ 0.1167, 0.1917, 0.1917, 0.1250, 0.1667, 0.1583 ,0.2194]
    ematro_reward = [ 0.1167, 0.1250, 0.1583, 0.1667, 0.1917, 0.1917, 0.2194]
    idrl_reward = [ 0.1500, 0.1583 ,0.1583, 0.1583, 0.2000,  0.2121, 0.2167]
    cdrl_reward = [ 0.1500, 0.1583, 0.1612, 0.1667, 0.1692 ,0.1785, 0.2000 ]

    ematro_migrations = [28, 25, 22, 21, 19, 19, 18]
    idrl_migrations = [ 27, 27, 25, 24, 23, 22, 22]
    cdrl_migrations = [27, 26, 24, 23, 22, 22, 17]


    #learning_rate = [1e-4, 5*1e-4, 1e-3, 5*1e-3]
    # ematro_energyconsumption = [ 802.5, 637.5, 637.5, 555.0]
    # idrl_energyconsumption = [ 1050.0, 885.0, 1050.0, 885.0]
    # cdrl_energyconsumption = [ 967.5, 885.0, 1050.0 ,1050.0]

    # ematro_reward = [ 0.1167, 0.1917, 0.1917, 0.2194]
    # idrl_reward = [ 0.1500, 0.2000, 0.1583, 0.1583]
    # cdrl_reward = [ 0.1785, 0.1500, 0.1612, 0.1583]

    # ematro_migrations = [21, 19, 19 ,18]
    # idrl_migrations = [ 25, 23, 27 ,22]
    # cdrl_migrations = [ 24, 27, 23 ,26]



    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.5))

    captions = {
    ax1: "(a) Average reward.",
    ax2: "(b) Average energy consumption.",
    ax3: "(c) Average migrations number.",
} 

    ax1.plot(learning_rate, ematro_reward, marker='o', color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")
    ax1.plot(learning_rate, idrl_reward, marker='o', color=COLORS["IDQN"], linewidth=1.8, label="IDQN")
    ax1.plot(learning_rate, cdrl_reward, marker='o', color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")
    ax1.set_xlabel("Learning rate", fontsize=11)
    ax1.set_ylabel("Average reward", fontsize=11)
    ax1.grid(True, alpha=0.3)


    ax2.plot(learning_rate, ematro_energyconsumption, marker='o', color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")
    ax2.plot(learning_rate, idrl_energyconsumption, marker='o', color=COLORS["IDQN"], linewidth=1.8, label="IDQN")
    ax2.plot(learning_rate, cdrl_energyconsumption, marker='o', color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")
    ax2.set_xlabel("Learning rate", fontsize=11)
    ax2.set_ylabel("Average energy consumption", fontsize=11)
    ax2.grid(True, alpha=0.3)

    ax3.plot(learning_rate, ematro_migrations, marker='o', color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")
    ax3.plot(learning_rate, idrl_migrations, marker='o', color=COLORS["IDQN"], linewidth=1.8, label="IDQN")
    ax3.plot(learning_rate, cdrl_migrations, marker='o', color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")
    ax3.set_xlabel("Learning rate", fontsize=11)
    ax3.set_ylabel("Average migrations number", fontsize=11)
    ax3.grid(True, alpha=0.3)

    # handles, labels = ax1.get_legend_handles_labels()


    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize='x-small',
    #  columnspacing=0.8, handlelength=1.2)
    # fig.subplots_adjust(top=0.85)

    offsets = [-0.04, 0.0, 0.04]  # shift left / none / right in figure fraction
    for (ax, cap), offset in zip(captions.items(), offsets):
        xlabel_bb = ax.xaxis.label.get_window_extent()
        xlabel_bb_fig = xlabel_bb.transformed(fig.transFigure.inverted())

        x_center = xlabel_bb_fig.x0 + xlabel_bb_fig.width / 2 + offset  #
        y_bottom  = xlabel_bb_fig.y0 - 0.05

        fig.text(x_center, y_bottom, cap,
                ha="center", va="top", fontsize=14, fontstyle="italic", transform=fig.transFigure)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=4, fontsize='small')
    
    fig.tight_layout()
    path = os.path.join(save_dir, "learning_rate_sensitivity_analysis.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")




def fig_sensitivity_analysis_discount_factor(save_dir):

    
    discount_factor = [0.80, 0.90, 0.95, 0.99 ] 

    ematro_reward = [ 0.1333, 0.1750 ,0.1333, 0.1917]
    idrl_reward = [ 0.1750, 0.1750 ,0.1917, 0.1583]
    cdrl_reward = [0.1917, 0.1500, 0.1583, 0.1612]

    ematro_energyconsumption = [ 802.5, 802.5, 885.0, 637.5]
    idrl_energyconsumption = [ 885.0, 885.0, 885.0,1050.0]
    cdrl_energyconsumption = [885.0, 1050.0, 1050.0, 1050.0]

    ematro_migrations = [ 24, 19, 25, 19]
    idrl_migrations = [ 25, 26, 27,27]
    cdrl_migrations = [23, 27, 26, 23]

    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.5))

    captions = {
    ax1: "(a) Average reward.",
    ax2: "(b) Average energy consumption.",
    ax3: "(c) Average migrations number.",
} 

    ax1.plot(discount_factor, ematro_reward, marker='o', color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")
    ax1.plot(discount_factor, idrl_reward, marker='o', color=COLORS["IDQN"], linewidth=1.8, label="IDQN")
    ax1.plot(discount_factor, cdrl_reward, marker='o', color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")
    ax1.set_xlabel("Discount factor", fontsize=11)
    ax1.set_ylabel("Average reward", fontsize=11)
    ax1.grid(True, alpha=0.3)


    ax2.plot(discount_factor, ematro_energyconsumption, marker='o', color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")
    ax2.plot(discount_factor, idrl_energyconsumption, marker='o', color=COLORS["IDQN"], linewidth=1.8, label="IDQN")
    ax2.plot(discount_factor, cdrl_energyconsumption, marker='o', color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")
    ax2.set_xlabel("Discount factor", fontsize=11)
    ax2.set_ylabel("Average energy consumption", fontsize=11)
    ax2.grid(True, alpha=0.3)

    ax3.plot(discount_factor, ematro_migrations, marker='o', color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")
    ax3.plot(discount_factor, idrl_migrations, marker='o', color=COLORS["IDQN"], linewidth=1.8, label="IDQN")
    ax3.plot(discount_factor, cdrl_migrations, marker='o', color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")
    ax3.set_xlabel("Discount factor", fontsize=11)
    ax3.set_ylabel("Average migrations number", fontsize=11)
    ax3.grid(True, alpha=0.3)


    offsets = [-0.04, 0.0, 0.04]  # shift left / none / right in figure fraction
    for (ax, cap), offset in zip(captions.items(), offsets):
        xlabel_bb = ax.xaxis.label.get_window_extent()
        xlabel_bb_fig = xlabel_bb.transformed(fig.transFigure.inverted())

        x_center = xlabel_bb_fig.x0 + xlabel_bb_fig.width / 2 + offset  #
        y_bottom  = xlabel_bb_fig.y0 - 0.05

        fig.text(x_center, y_bottom, cap,
                ha="center", va="top", fontsize=14, fontstyle="italic", transform=fig.transFigure)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=4, fontsize='small')
    
    fig.tight_layout()
    path = os.path.join(save_dir, "discount_sensitivity_analysis.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")



def fig_delay_curve(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep  = df_e_matro["episode"].values
    raw = df_e_matro["avg_delay_violation"].values
    ax.plot(ep, raw, alpha=0.2, color=COLORS["E-MATRO++"], linewidth=0.6)
    ax.plot(ep, smooth(raw), color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    ep2  = df_dqn["episode"].values
    raw2 = df_dqn["avg_delay_violation"].values
    ax.plot(ep2, raw2, alpha=0.2, color=COLORS["C-DQN"], linewidth=0.6)
    ax.plot(ep2, smooth(raw2), color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    ep3  = df_idqn["episode"].values
    raw3 = df_idqn["avg_delay_violation"].values
    ax.plot(ep3, raw3, alpha=0.2, color=COLORS["IDQN"], linewidth=0.6)
    ax.plot(ep3, smooth(raw3), color=COLORS["IDQN"], linewidth=1.8, label="IDQN")


    ep4 = df_random["episode"].values
    raw4 = df_random["avg_delay_violation"].values
    ax.plot(ep4, raw4, alpha=0.2, color=COLORS["Random"], linewidth=0.6)
    ax.plot(ep4, smooth(raw4), color=COLORS["Random"], linewidth=1.8, label="Random")

    ep5 = df_group_all["episode"].values
    raw5 = df_group_all["avg_delay_violation"].values
    ax.plot(ep5, raw5, alpha=0.2, color=COLORS["GroupAll"], linewidth=0.6)
    ax.plot(ep5, smooth(raw5), color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    ep6 = df_split_all["episode"].values
    raw6 = df_split_all["avg_delay_violation"].values
    ax.plot(ep6, raw6, alpha=0.2, color=COLORS["SplitAll"], linewidth=0.6)
    ax.plot(ep6, smooth(raw6), color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")

    

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Average delay (ms)", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "delay_curve.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_cdf_delay_rate(df_e_matro, df_dqn, df_idqn, df_random, df_group_all, df_split_all, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3.5))


    cr_ematro = np.sort(df_e_matro["avg_delay_violation"].values)
    cdf_ma     = np.arange(1, len(cr_ematro) + 1) / len(cr_ematro)
    ax.plot(cr_ematro, cdf_ma, color=COLORS["E-MATRO++"], linewidth=1.8, label="E-MATRO++")

    cr_dqn = np.sort(df_dqn["avg_delay_violation"].values)
    cdf_dqn     = np.arange(1, len(cr_dqn) + 1) / len(cr_dqn)
    ax.plot(cr_dqn, cdf_dqn, color=COLORS["C-DQN"], linewidth=1.8, label="C-DQN")

    cr_idqn = np.sort(df_idqn["avg_delay_violation"].values)
    cdf_idqn     = np.arange(1, len(cr_idqn) + 1) / len(cr_idqn)
    ax.plot(cr_idqn, cdf_idqn, color=COLORS["IDQN"], linewidth=1.8, label="IDQN")


    cr_random = np.sort(df_random["avg_delay_violation"].values)
    cdf_random     = np.arange(1, len(cr_random) + 1) / len(cr_random)
    ax.plot(cr_random, cdf_random, color=COLORS["Random"], linewidth=1.8, label="Random") 

    cr_group_all = np.sort(df_group_all["avg_delay_violation"].values)
    cdf_group_all     = np.arange(1, len(cr_group_all) + 1) / len(cr_group_all)
    ax.plot(cr_group_all, cdf_group_all, color=COLORS["GroupAll"], linewidth=1.8, label="GroupAll")

    cr_split_all = np.sort(df_split_all["avg_delay_violation"].values)
    cdf_split_all     = np.arange(1, len(cr_split_all) + 1) / len(cr_split_all)
    ax.plot(cr_split_all, cdf_split_all, color=COLORS["SplitAll"], linewidth=1.8, label="SplitAll")


    ax.set_xlabel("Average delay (ms)", fontsize=11)
    ax.set_ylabel("Empirical CDF", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize='x-small',
     columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "cdf_avg_delay.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")













def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="data_comparison/results_4rus_4nodes")
    p.add_argument("--save-dir", default=None)
    args = p.parse_args()

    save_dir = args.save_dir or args.results_dir

    figures_dir = os.path.join(save_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    
    # Training logs
    e_matro_csv = os.path.join(args.results_dir, "e-matro_training_log_4rus_4nodes_10000_episodes.csv")
    dqn_csv   = os.path.join(args.results_dir, "ctde_training_log_4rus_4nodes_10000_episodes.csv")
    idqn_csv   = os.path.join(args.results_dir, "idqn_training_log_4rus_4nodes_10000_episodes.csv")

    random_csv = os.path.join(args.results_dir, "random_training_log_4rus_4nodes_10000_episodes.csv")
    group_all_csv = os.path.join(args.results_dir, "group_all_training_log_4rus_4nodes_10000_episodes.csv")
    split_all_csv = os.path.join(args.results_dir, "split_all_training_log_4rus_4nodes_10000_episodes.csv")

    # for name, df in [('E-MATRO', e_matro_csv), ('C-DQN', dqn_csv), ('IDQN', idqn_csv)]:
    #     last = df.tail(500)
    #     print(f"{name}:")
    #     print(f"  Energy efficiency : {last['energy_efficiency'].mean():.3f}")
    #     print(f"  Energy saving     : {last['energy_saving_%'].mean():.1f}%")
    #     print(f"  Migrations/ep     : {last['total_migrations'].mean():.1f}")
    #     print(f"  OCloud util       : {last['ocloud_utilization'].mean():.3f}")

   
    if not os.path.exists(e_matro_csv):
        print(f"E-MATRO training log not found: {e_matro_csv}")
        return

    if not os.path.exists(dqn_csv):
        print(f"C-DQN training log not found: {dqn_csv}")
        return

    if not os.path.exists(idqn_csv):
        print(f"IDQN training log not found: {idqn_csv}")
        return

    if not os.path.exists(random_csv):
        print(f"Random training log not found: {random_csv}")
        return  

    if not os.path.exists(group_all_csv):
        print(f"GroupAll training log not found: {group_all_csv}")
        return
    
    if not os.path.exists(split_all_csv):
        print(f"SplitAll training log not found: {split_all_csv}")
        return

    # Read CSV files
    df_e_matro_csv = pd.read_csv(e_matro_csv)
    df_dqn = pd.read_csv(dqn_csv)
    df_idqn = pd.read_csv(idqn_csv)
    df_random = pd.read_csv(random_csv)
    df_group_all = pd.read_csv(group_all_csv)
    df_split_all = pd.read_csv(split_all_csv)

    print(f"E-MATRO: {len(df_e_matro_csv)} episodes")
    print(f"C-DQN : {len(df_dqn)} episodes")
    print(f"IDQN : {len(df_idqn)} episodes")

    print(f"Random : {len(df_random)} episodes")
    print(f"GroupAll : {len(df_group_all)} episodes")
    print(f"SplitAll : {len(df_split_all)} episodes")



    # Generate figure
    fig1_reward_curve(df_e_matro_csv, df_dqn, df_idqn,  figures_dir)#df_random, df_group_all, df_split_all,
    fig_cdf_reward(df_e_matro_csv, df_dqn, df_idqn,  figures_dir) # df_random, df_group_all, df_split_all,
    fig_ccdf_reward(df_e_matro_csv, df_dqn, df_idqn,  figures_dir)
    fig2_energy_curve(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig3_migrations_curve(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig4_loss_curve(df_e_matro_csv, df_dqn, df_idqn,  figures_dir)
    fig5_conflicts_curve(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig6_nodes_curve(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig8_energy_saving_curve(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig9_inference_curve(df_e_matro_csv, df_dqn, df_idqn, figures_dir)
    #fig10_training_curve(df_e_matro_csv, df_dqn, df_idqn, figures_dir)
    fig11_peak_load_curve(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig12_energy_efficiency_curve(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig13_ocloud_util_curve(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig14_capacity_violations_curve(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig15_conflict_rate_curve(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig16_migration_success_rate_curve(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig17_OCloud_bar_chart(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig_cdf_CloudUtil(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig_cdf_migration(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig_bar_chart_migration(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig_cdf_capacity_violations(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig_cdf_conflict_rate(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig_cdf_inference_time(df_e_matro_csv, df_dqn, df_idqn, figures_dir)
    fig_delay_curve(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)
    fig_cdf_delay_rate(df_e_matro_csv, df_dqn, df_idqn, df_random, df_group_all, df_split_all, figures_dir)


    #sensitivity analysis 
    fig_energy_eff_learning_rate(figures_dir)
    fig_reward_learning_rate(figures_dir)
    fig_migration_learning_rate(figures_dir)

    fig_sensitivity_analysis_learning_rate(figures_dir)
    fig_sensitivity_analysis_discount_factor(figures_dir)


if __name__ == "__main__":
    main()