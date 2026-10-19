"""
plots_sensitivity.py — Generate paper-ready figures from multi-agent deep reinforcement learning results.

Figures produced:
  1. reward_curve.png       — learning curve (avg reward per episode)
  2. energy_curve.png       — episode energy over training
  3. constraint_scores.png  — constraint satisfaction rates over training
  4. energy_comparison.png  — bar chart: DQN vs baselines
  5. migration_comparison.png — migration count comparison
  6. platform_vs_sim.png    — simulation vs real platform energy (if both present)

Usage:
  python3 plots_sensitivity.py --results-dir sensitivity_analysis_logs/ --save-dir figures/
  python3 plots_sensitivity.py --results-dir /sensitivity_analysis_logs/ --save-dir figures/
"""



import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path

plt.style.use('/home/ubuntu/E-matro-evaluation/academic_paper.mplstyle')



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


def smooth(values, window=50):
    kernel = np.ones(window) / window
    pad    = np.pad(values, (window//2, window//2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")[:len(values)]



def fig1_reward_anal_lr(df1, df2, df3, df7,  save_dir): 
    fig, ax = plt.subplots(figsize=(7, 3.5))

    ep1  = df1["episode"].values
    raw = df1["avg_reward"].values
    ax.plot(ep1, smooth(raw), color="C1", linewidth=1.8, label=r'$\alpha=1e-4$')

    ep2 = df2["episode"].values
    raw2 = df2["avg_reward"].values
    ax.plot(ep2, smooth(raw2), color="C2", linewidth=1.8, label=r'$\alpha=5*1e-4$')

    ep3 = df3["episode"].values
    raw3 = df3["avg_reward"].values
    ax.plot(ep3, smooth(raw3), color="C3", linewidth=1.8, label=r'$\alpha=1e-3$')


    ep7 = df7["episode"].values
    raw7 = df7["avg_reward"].values
    ax.plot(ep7, smooth(raw7), color="C4", linewidth=1.8, label=r'$\alpha=5*1e-3$')

  

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Average reward", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "ematro_reward_lr_analysis.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")   



def fig2_energy_anal_lr(df1, df2, df3, df7,  save_dir): 
    fig, ax = plt.subplots(figsize=(7, 3.5))

    ep1  = df1["episode"].values
    raw = df1["avg_energy"].values
    ax.plot(ep1, smooth(raw), color="C1", linewidth=1.8, label=r'$\alpha=1e-4$')

    ep2 = df2["episode"].values
    raw2 = df2["avg_energy"].values
    ax.plot(ep2, smooth(raw2), color="C2", linewidth=1.8, label=r'$\alpha=5*1e-4$')

    ep3 = df3["episode"].values
    raw3 = df3["avg_energy"].values
    ax.plot(ep3, smooth(raw3), color="C3", linewidth=1.8, label=r'$\alpha=1e-3$')


    ep7 = df7["episode"].values
    raw7 = df7["avg_energy"].values
    ax.plot(ep7, smooth(raw7), color="C4", linewidth=1.8, label=r'$\alpha=5*1e-3$')

  

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode energy (kWh)", fontsize=11)
    
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "ematro_energy_lr_analysis.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")   



def fig3_migration_anal_lr(df1, df2, df3, df7,  save_dir): 
    fig, ax = plt.subplots(figsize=(7, 3.5))

    ep1  = df1["episode"].values
    raw = df1["total_migrations"].values
    ax.plot(ep1, smooth(raw), color="C1", linewidth=1.8, label=r'$\alpha=1e-4$')

    ep2 = df2["episode"].values
    raw2 = df2["total_migrations"].values
    ax.plot(ep2, smooth(raw2), color="C2", linewidth=1.8, label=r'$\alpha=5*1e-4$')

    ep3 = df3["episode"].values
    raw3 = df3["total_migrations"].values
    ax.plot(ep3, smooth(raw3), color="C3", linewidth=1.8, label=r'$\alpha=1e-3$')


    ep7 = df7["episode"].values
    raw7 = df7["total_migrations"].values
    ax.plot(ep7, smooth(raw7), color="C4", linewidth=1.8, label=r'$\alpha=5*1e-3$')

  

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Number of migrations", fontsize=11)
    
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "ematro_migrations_lr_analysis.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")




def fig4_reward_anal_gamma(df1, df2, df3, df4,  save_dir): 
    fig, ax = plt.subplots(figsize=(7, 3.5))

    ep1  = df1["episode"].values
    raw = df1["avg_reward"].values
    ax.plot(ep1, smooth(raw), color="C1", linewidth=1.8, label=r'$\gamma=0.8$')

    ep2 = df2["episode"].values
    raw2 = df2["avg_reward"].values
    ax.plot(ep2, smooth(raw2), color="C2", linewidth=1.8, label=r'$\gamma=0.90$')

    ep3 = df3["episode"].values
    raw3 = df3["avg_reward"].values
    ax.plot(ep3, smooth(raw3), color="C3", linewidth=1.8, label=r'$\gamma=0.95$')


    ep4 = df4["episode"].values
    raw4 = df4["avg_reward"].values
    ax.plot(ep4, smooth(raw4), color="C4", linewidth=1.8, label=r'$\gamma=0.99$')

  

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Average reward", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "ematro_reward_gamma_analysis.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}") 



def fig5_energy_anal_gamma(df1, df2, df3, df4, save_dir): 
    fig, ax = plt.subplots(figsize=(7, 3.5))

    ep1  = df1["episode"].values
    raw = df1["avg_energy"].values
    ax.plot(ep1, smooth(raw), color="C1", linewidth=1.8, label=r'$\gamma=0.80$')

    ep2 = df2["episode"].values
    raw2 = df2["avg_energy"].values
    ax.plot(ep2, smooth(raw2), color="C2", linewidth=1.8, label=r'$\gamma=0.90$')

    ep3 = df3["episode"].values
    raw3 = df3["avg_energy"].values
    ax.plot(ep3, smooth(raw3), color="C3", linewidth=1.8, label=r'$\gamma=0.95$')


    ep4 = df4["episode"].values
    raw4 = df4["avg_energy"].values
    ax.plot(ep4, smooth(raw4), color="C4", linewidth=1.8, label=r'$\gamma=0.99$')

  

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode energy (kWh)", fontsize=11)
    
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "ematro_energy_gamma_analysis.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")   



def fig6_migration_anal_gamma(df1, df2, df3, df4,  save_dir): 
    fig, ax = plt.subplots(figsize=(7, 3.5))

    ep1  = df1["episode"].values
    raw = df1["total_migrations"].values
    ax.plot(ep1, smooth(raw), color="C1", linewidth=1.8, label=r'$\gamma=0.8$')

    ep2 = df2["episode"].values
    raw2 = df2["total_migrations"].values
    ax.plot(ep2, smooth(raw2), color="C2", linewidth=1.8, label=r'$\gamma=0.90$')

    ep3 = df3["episode"].values
    raw3 = df3["total_migrations"].values
    ax.plot(ep3, smooth(raw3), color="C3", linewidth=1.8, label=r'$\gamma=0.95$')


    ep4 = df4["episode"].values
    raw4 = df4["total_migrations"].values
    ax.plot(ep4, smooth(raw4), color="C4", linewidth=1.8, label=r'$\gamma=0.99$')


    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Number of migrations", fontsize=11)
    
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "ematro_migrations_gamma_analysis.png")
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")





def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="sensitivity_analysis_logs")
    p.add_argument("--save-dir", default=None)
    args = p.parse_args()

    save_dir = args.save_dir or args.results_dir

    figures_dir = os.path.join(save_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    
    # Training logs
    lr1_csv = os.path.join(args.results_dir, "e-matro_training_log_9rus_6nodes_lr_1e-4.csv")
    lr2_csv   = os.path.join(args.results_dir, "e-matro_training_log_9rus_6nodes_lr_5*1e-4.csv")
    lr3_csv   = os.path.join(args.results_dir, "e-matro_training_log_9rus_6nodes_lr_1e-3.csv")

    lr4_csv = os.path.join(args.results_dir, "e-matro_training_log_9rus_6nodes_lr_0.002.csv")
    lr5_csv = os.path.join(args.results_dir, "e-matro_training_log_9rus_6nodes_lr_0.003.csv")
    lr6_csv = os.path.join(args.results_dir, "e-matro_training_log_9rus_6nodes_lr_0.004.csv")
    lr7_csv   = os.path.join(args.results_dir, "e-matro_training_log_9rus_6nodes_lr_5*1e-3.csv")


    gamma1_csv = os.path.join(args.results_dir, "e-matro_training_log_9rus_6nodes_gamma_0.80.csv")
    gamma2_csv = os.path.join(args.results_dir, "e-matro_training_log_9rus_6nodes_gamma_0.90.csv")
    gamma3_csv = os.path.join(args.results_dir, "e-matro_training_log_9rus_6nodes_gamma_0.95.csv")
    gamma4_csv = os.path.join(args.results_dir, "e-matro_training_log_9rus_6nodes_gamma_0.99.csv")


    df_lr1 = pd.read_csv(lr1_csv)
    df_lr2 = pd.read_csv(lr2_csv)
    df_lr3 = pd.read_csv(lr3_csv)
    df_lr4 = pd.read_csv(lr4_csv)
    df_lr5 = pd.read_csv(lr5_csv)
    df_lr6 = pd.read_csv(lr6_csv)
    df_lr7 = pd.read_csv(lr7_csv)


    df_gamma1 = pd.read_csv(gamma1_csv)
    df_gamma2 = pd.read_csv(gamma2_csv)
    df_gamma3 = pd.read_csv(gamma3_csv)
    df_gamma4 = pd.read_csv(gamma4_csv)


    #generate figures
    fig1_reward_anal_lr(df_lr1, df_lr2, df_lr3, df_lr7, figures_dir)
    fig2_energy_anal_lr(df_lr1, df_lr2, df_lr3, df_lr7, figures_dir)
    fig3_migration_anal_lr(df_lr1, df_lr2, df_lr3, df_lr7, figures_dir)


    fig4_reward_anal_gamma(df_gamma1, df_gamma2, df_gamma3, df_gamma4, figures_dir)
    fig5_energy_anal_gamma(df_gamma1, df_gamma2, df_gamma3, df_gamma4, figures_dir)
    fig6_migration_anal_gamma(df_gamma1, df_gamma2, df_gamma3, df_gamma4, figures_dir)



if __name__ == "__main__":
  main()