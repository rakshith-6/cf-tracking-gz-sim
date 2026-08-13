#!/usr/bin/env python3
"""
Analyze the experiments by visualization with plots:

Usage:
    python3 plot_tracking.py <run_dir> [<run_dir2> ...]

- Each run_dir is a directory produced by cf_tracking/harness_flight.py
  which has log.csv, meta.json and metrics.json 
- With one run_dir: writes tracking_plots.png into it (Six subplots: 2D 
  tracking, altitude vs t, per axis position error, velocity error, 
  attitude/yaw error and rotor commands)
- With several run_dir: writes comparison.png(4 overlay subplots labeled 
  by controller: 2D tracking, pos, vel and attitude error) into the 
  first run's parent directory.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_run(run_dir):
    '''
    Given a run_dir the fun loads all the experiment data(loads log.csv, 
    meta.json and metrics.json)
    '''
    df = pd.read_csv(os.path.join(run_dir, 'log.csv'))
    with open(os.path.join(run_dir, 'meta.json')) as f:
        meta = json.load(f)
    mpath = os.path.join(run_dir, 'metrics.json')
    metrics = {}
    if os.path.exists(mpath):
        with open(mpath) as f:
            metrics = json.load(f)
    return df, meta, metrics

def plot_single(run_dir):
    '''
    Plots results of one experiment
    '''
    df, meta, metrics = load_run(run_dir)
    track = df[df.phase.isin(['primitive', 'hover'])]
    t0 = track.t.iloc[0] if len(track) else df.t.iloc[0]
    tt = df.t - t0

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    label = f"{meta.get('controller')} / {meta.get('primitive')}" \
        if meta.get('mode') == 'track' else \
        f"{meta.get('controller')} / {meta.get('mode')}"
    sub = []
    if metrics.get('pos_rmse') is not None:
        sub.append(f"pos RMSE {metrics['pos_rmse']*100:.1f} cm "
                   f"(max {metrics['pos_err_max']*100:.1f} cm)")
    if metrics.get('mean_rtf') is not None:
        sub.append(f"RTF {metrics['mean_rtf']:.2f}")
    fig.suptitle(f"{os.path.basename(run_dir.rstrip('/'))} — {label}"
                 + ('\n' + ', '.join(sub) if sub else ''))

    # 2D trajectory tracking in XY plane(track phase only)
    ax = axes[0, 0]
    ax.plot(track.ref_px, track.ref_py, 'k--', lw=1, label='reference')
    ax.plot(track.px, track.py, lw=1, label='actual')
    ax.set_xlabel('x[m]'); ax.set_ylabel('y[m]')
    ax.set_title('2D trajectory tracking(track phase)')
    ax.axis('equal'); ax.legend()

    # z vs t(all phases)
    ax = axes[0, 1]
    ax.plot(tt, df.ref_pz, 'k--', lw=1, label='z_ref')
    ax.plot(tt, df.pz, lw=1, label='z')
    for ph in df.phase.unique():
        i = df.index[df.phase == ph][0]
        ax.axvline(tt.iloc[i] if hasattr(tt, 'iloc') else tt[i],
                   color='gray', lw=0.5, alpha=0.5)
    ax.set_xlabel('t[s]'); ax.set_ylabel('z [m]')
    ax.set_title('Altitude(phase boundaries in gray)'); ax.legend()

    # Per axis position error plot
    ax = axes[1, 0]
    for c, lab in (('err_px', 'x'), ('err_py', 'y'), ('err_pz', 'z')):
        ax.plot(tt, df[c] * 100, lw=0.8, label=lab)
    ax.plot(tt, df.err_pos * 100, 'k', lw=1, label='|err|')
    ax.set_xlabel('t[s]'); ax.set_ylabel('position error[cm]')
    ax.set_title('Position error'); ax.legend()

    # Velocity error
    ax = axes[1, 1]
    ax.plot(tt, df.err_vel, lw=0.8)
    ax.set_xlabel('t[s]'); ax.set_ylabel('|velocity error|[m/s]')
    ax.set_title('Velocity error')

    # Attitude / Yaw error
    ax = axes[2, 0]
    ax.plot(tt, df.att_err_deg, lw=0.8, label='thrust axis err[deg]')
    ax.plot(tt, np.degrees(df.err_yaw), lw=0.8, label='yaw err[deg]')
    ax.plot(tt, df.tilt_deg, lw=0.6, alpha=0.6, label='tilt[deg]')
    ax.set_xlabel('t [s]'); ax.set_ylabel('[deg]')
    ax.set_title('attitude'); ax.legend()

    # Rotor commands
    ax = axes[2, 1]
    for i in range(1, 5):
        ax.plot(tt, df[f'w{i}'], lw=0.6, label=f'w{i}')
    wmax = meta.get('vehicle', {}).get('motor_speed_max')
    if wmax:
        ax.axhline(wmax, color='r', ls=':', lw=1, label='max')
    ax.set_xlabel('t[s]'); ax.set_ylabel('rotor speed[rad/s]')
    ax.set_title(f"rotor commands "
                 f"(saturated {100*df.saturated.mean():.1f}% of ticks)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    
    out = os.path.join(run_dir, 'tracking_plots.png')
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')
    if metrics:
        print(json.dumps(metrics, indent=2))

def plot_compare(run_dirs):
    '''
    Comparison plot of tracking from multiple experiment or multiple 
    controllers
    '''
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Controller comparison(track phase)')
    rows = []
    for rd in run_dirs:
        df, meta, metrics = load_run(rd)
        track = df[df.phase.isin(['primitive', 'hover'])]
        lab = f"{meta.get('controller')} ({os.path.basename(rd.rstrip('/'))})"
        tt = track.t - track.t.iloc[0]
        axes[0, 0].plot(track.px, track.py, lw=1, label=lab)
        axes[0, 1].plot(tt, track.err_pos * 100, lw=0.8, label=lab)
        axes[1, 0].plot(tt, track.err_vel, lw=0.8, label=lab)
        axes[1, 1].plot(tt, track.att_err_deg, lw=0.8, label=lab)
    
        rows.append((lab, metrics))
        if rd == run_dirs[0]:
            axes[0, 0].plot(track.ref_px, track.ref_py, 'k--', lw=1,
                            label='reference')
    
    axes[0, 0].set_title('2D tracking'); axes[0, 0].axis('equal')
    axes[0, 1].set_title('|position error|[cm]')
    axes[1, 0].set_title('|velocity error|[m/s]')
    axes[1, 1].set_title('thrust axis error[deg]')
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.set_xlabel('t[s]' if ax is not axes[0, 0] else 'x[m]')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(run_dirs[0].rstrip('/')),
                       'comparison.png')
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')
    for lab, m in rows:
        print(f"{lab}: pos_rmse={m.get('pos_rmse')}, "
              f"max={m.get('pos_err_max')}, sat={m.get('saturated_frac')}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if len(sys.argv) == 2:
        plot_single(sys.argv[1])
    else:
        plot_compare(sys.argv[1:])
