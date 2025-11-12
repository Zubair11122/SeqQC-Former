#!/usr/bin/env python3
"""
SeqQC-Former vs Mutect2 & Strelka2 — Publication-Ready Comparison (PNG only)
Updated: Memory-optimized version with reduced DPI for large datasets
"""

from pathlib import Path
import argparse, sys
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss, roc_auc_score
)
from sklearn.calibration import calibration_curve

# ---------- Optional venn ----------
try:
    from matplotlib_venn import venn2, venn3
    HAS_VENN = True
except Exception:
    HAS_VENN = False
    venn2 = venn3 = None

# ---------- Optional YAML ----------
try:
    import yaml
except Exception:
    yaml = None  # handled below

# ===================== UPDATED Visualization Settings =====================
FONT_FAMILY = "Arial"  # More professional font
mpl.rcParams.update({
    'font.family':       FONT_FAMILY,
    'font.size':         10,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'figure.dpi':        300,  # Reduced from 900 to prevent memory issues
    'savefig.dpi':       300,  # Reduced from 900
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'grid.linewidth':    0,  # Remove grid
})

# Clean style - no grids
plt.style.use('default')
sns.set_theme(style="white", font_scale=1.0)  # Changed from whitegrid to white

# Consistent color scheme for all tools across all plots
TOOL_COLORS = {
    'SeqQC-Former': '#2E8B57',  # Sea Green - consistent primary color
    'Mutect2':      '#4169E1',  # Royal Blue  
    'Strelka2':     '#DC143C',  # Crimson Red
}

# Class colors for distributions
CLASS_COLORS = {
    0: '#7F7F7F',  # Gray for negative class
    1: '#2E8B57',  # Same green as SeqQC-Former for positive class
}

LINEWIDTH = 2.0
MARKER_SIZE = 6

COL_SYNONYMS = {
    "chrom": ["chrom", "chromosome", "#chrom", "chr", "CHROM", "Chromosome"],
    "pos"  : ["pos", "position", "start", "start_position", "bp", "start_bp", "POS", "Start_Position"],
    "ref"  : ["ref", "reference", "ref_allele", "reference_allele", "REF", "Reference_Allele"],
    "alt"  : ["alt", "alt_allele", "alternate", "ALT", "ALT1", "Tumor_Seq_Allele2"],
    "qual" : ["qual", "QUAL"],
}

# ===================== Small utilities =====================
def find_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols: return cols[n.lower()]
    for c in df.columns:
        cl = c.lower()
        for n in names:
            if n.lower() in cl: return c
    return None

def ensure_key(df: pd.DataFrame) -> pd.DataFrame:
    if 'key' in df.columns: return df
    c = find_col(df, COL_SYNONYMS["chrom"])
    p = find_col(df, COL_SYNONYMS["pos"])
    r = find_col(df, COL_SYNONYMS["ref"])
    a = find_col(df, COL_SYNONYMS["alt"])
    if not all([c,p,r,a]):
        raise ValueError("Cannot construct 'key': need chrom/pos/ref/alt.")
    df = df.copy()
    df['key'] = (
        df[c].astype(str).str.replace(r'^chr','',regex=True) + ":" +
        df[p].astype(str) + ":" +
        df[r].astype(str).str.split('[,;]').str[0].str.upper() + ":" +
        df[a].astype(str).str.split('[,;]').str[0].str.upper()
    )
    return df

# ---------------------- ensure_prob (robust) ----------------------
def ensure_prob(df: pd.DataFrame, tool_name: str) -> pd.DataFrame:
    """
    Build a probability-like score for a caller row.
    Preference:
      Mutect2:   TLOD -> AF (explicit or from depths) -> QUAL
      Strelka2:  SomaticEVS -> AF (explicit or from depths) -> QUAL
      Generic:   prob/score -> AF (explicit or depths) -> QUAL
    Robust fallbacks:
      - If AF not present, compute from depth columns (e.g., AD or t_ref_count/t_alt_count)
      - If still constant/zero, use presence/FILTER as a last-resort discriminator
    """
    df = df.copy()
    low = {c.lower(): c for c in df.columns}
    def has(c): return c.lower() in low
    def col(c): return low[c.lower()]

    def _minmax01_series(x):
        v = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
        vmin, vmax = v.min(skipna=True), v.max(skipna=True)
        if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
            return v.fillna(0.0).replace(np.nan, 0.0)
        return (v - vmin) / (vmax - vmin)

    # ---------- Try to build AF if we can ----------
    af_series = None
    for af_name in ["AF", "af", "TumorAF", "tumor_af", "FA", "fa"]:
        if has(af_name):
            af_series = pd.to_numeric(df[col(af_name)], errors="coerce")
            break

    if af_series is None:
        # separate tumor ref/alt counts
        ref_cands = [n for n in low if n in {"t_ref_count","tumor_ref_count","tref","ref_count"}]
        alt_cands = [n for n in low if n in {"t_alt_count","tumor_alt_count","talt","alt_count"}]
        if ref_cands and alt_cands:
            tref = pd.to_numeric(df[low[ref_cands[0]]], errors="coerce").clip(lower=0)
            talt = pd.to_numeric(df[low[alt_cands[0]]], errors="coerce").clip(lower=0)
            denom = (tref + talt).replace(0, np.nan)
            af_series = (talt / denom).fillna(0.0)

    if af_series is None:
        # AD-like "ref,alt"
        for ad_name in ["AD", "ad", "TUMOR_AD", "tumor_ad"]:
            if ad_name.lower() in low:
                parts = df[col(ad_name)].astype(str).str.split("[,:;]", regex=True)
                try:
                    t_ref = pd.to_numeric(parts.str[0], errors="coerce").clip(lower=0)
                    t_alt = pd.to_numeric(parts.str[1], errors="coerce").clip(lower=0)
                    denom = (t_ref + t_alt).replace(0, np.nan)
                    af_series = (t_alt / denom).fillna(0.0)
                    break
                except Exception:
                    pass

    chosen = None
    if tool_name == "Mutect2":
        if has("TLOD"):
            chosen = _minmax01_series(df[col("TLOD")])
        elif af_series is not None:
            chosen = af_series.clip(0,1)
        elif has("QUAL"):
            q = pd.to_numeric(df[col("QUAL")], errors="coerce")
            chosen = 1.0/(1.0+np.exp(-(q/10.0)))
    elif tool_name == "Strelka2":
        if has("SomaticEVS"):
            chosen = _minmax01_series(df[col("SomaticEVS")])
        elif af_series is not None:
            chosen = af_series.clip(0,1)
        elif has("QUAL"):
            q = pd.to_numeric(df[col("QUAL")], errors="coerce")
            chosen = 1.0/(1.0+np.exp(-(q/10.0)))
    else:
        for cand in ["prob","score"]:
            if has(cand):
                chosen = pd.to_numeric(df[col(cand)], errors="coerce")
                break
        if chosen is None and af_series is not None:
            chosen = af_series.clip(0,1)
        if chosen is None and has("QUAL"):
            q = pd.to_numeric(df[col("QUAL")], errors="coerce")
            chosen = 1.0/(1.0+np.exp(-(q/10.0)))

    if chosen is None:
        chosen = pd.Series(0.0, index=df.index)
    chosen = pd.to_numeric(chosen, errors="coerce").fillna(0.0)

    # If constant, presence/FILTER fallback
    if chosen.nunique(dropna=False) <= 1:
        if has("FILTER"):
            filt = df[col("FILTER")].astype(str).str.upper().fillna("")
            presence_score = np.where(filt == "PASS", 1.0, 0.5)
        elif "filter" in low:
            filt = df[low["filter"]].astype(str).str.upper().fillna("")
            presence_score = np.where(filt == "PASS", 1.0, 0.5)
        else:
            presence_score = np.ones(len(df), dtype=float)
        chosen = pd.Series(presence_score, index=df.index)

    df["prob"] = chosen.clip(0.0, 1.0)
    return df

# ===================== Config & path resolution =====================
def load_yaml_config(path: Optional[Path]) -> dict:
    if path is None: return {}
    if not path.exists():
        print(f"[WARN] config.yaml not found at {path}; continuing with defaults.", file=sys.stderr)
        return {}
    if yaml is None:
        raise ImportError("PyYAML not installed. Install: python -m pip install pyyaml")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must parse to a mapping.")
    return data

def resolve_paths(args) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_cfg = script_dir / "config.yaml"
    cfg_path = args.config if args.config is not None else (default_cfg if default_cfg.exists() else None)
    cfg = load_yaml_config(cfg_path)

    data_root = Path(args.data_root) if args.data_root else \
                Path(cfg.get("data_root")) if cfg.get("data_root") else \
                (Path.home() / "Project" / "data_root")

    seqqc_default = None
    for candidate in ["full_preds_clean_by_key.csv","full_preds_with_keys.csv","full_preds.csv"]:
        cand_path = data_root / candidate
        if cand_path.exists():
            seqqc_default = cand_path; break
    if seqqc_default is None:
        seqqc_default = data_root / "full_preds.csv"

    seqqc  = Path(args.seqqc)  if args.seqqc  else Path(cfg.get("seqqc_csv"))  if cfg.get("seqqc_csv")  else seqqc_default
    mutect = Path(args.mutect) if args.mutect else Path(cfg.get("mutect_csv")) if cfg.get("mutect_csv") else (data_root / "baseline_out" / "mutect2_IL_1_bwa.csv")
    strelka= Path(args.strelka)if args.strelka else Path(cfg.get("strelka_csv"))if cfg.get("strelka_csv")else (data_root / "baseline_out" / "strelka2_IL_1_bwa.csv")
    outdir = Path(args.outdir) if args.outdir else Path(cfg.get("outdir")) if cfg.get("outdir") else (data_root / "baseline_out" / "comparison_results")

    args.data_root = data_root
    args.seqqc = seqqc
    args.mutect = mutect
    args.strelka = strelka
    args.outdir = outdir
    args.config = cfg_path
    return args

# ===================== Truth loading =====================
def load_truth_df(data_root: Path) -> pd.DataFrame:
    csv_path = data_root / "variants_labeled.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Missing {csv_path}.")
    df = pd.read_csv(csv_path)

    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("chromosome","chrom","chr","#chrom"): ren[c] = "chrom"
        elif lc in ("start_position","pos","position","start"): ren[c] = "pos"
        elif lc in ("reference_allele","ref","reference"): ren[c] = "ref"
        elif lc in ("tumor_seq_allele2","alt","alt_allele"): ren[c] = "alt"
        elif lc in ("seqc2_positive","label","y"): ren[c] = "label"
    df = df.rename(columns=ren)

    need = {"chrom","pos","ref","alt","label"}
    miss = need - set(df.columns)
    if miss: raise ValueError(f"{csv_path.name} missing columns: {miss}")

    df["key"] = (
        df["chrom"].astype(str).str.replace(r'^chr','',regex=True) + ":" +
        df["pos"].astype(str) + ":" +
        df["ref"].astype(str).str.split('[,;]').str[0].str.upper() + ":" +
        df["alt"].astype(str).str.split('[,;]').str[0].str.upper()
    )
    df["label"] = df["label"].astype(int)
    return df[["key","label"]].drop_duplicates("key")

def _class_counts(df: pd.DataFrame, name: str) -> tuple[int,int]:
    pos = int((df['label'].astype(int) == 1).sum())
    neg = int((df['label'].astype(int) == 0).sum())
    print(f"[DIAG] {name}: N={len(df)}  positives={pos}  negatives={neg}")
    return pos, neg

# ===================== Align to truth universe =====================
def align_to_truth_universe(truth_df: pd.DataFrame, tool_df: pd.DataFrame, tool_name: str) -> pd.DataFrame:
    tool_df = tool_df.copy()
    tool_df = tool_df[['key','prob']].drop_duplicates('key')
    out = truth_df[['key','label']].merge(tool_df, on='key', how='left')
    out['prob'] = pd.to_numeric(out['prob'], errors='coerce').fillna(0.0).clip(0.0, 1.0)
    out['TOOL'] = tool_name
    return out[['key','label','prob','TOOL']]

# ===================== Metrics helpers =====================
def precision_at_recall(y_true, y_score, target_recall=0.95):
    p, r, t = precision_recall_curve(y_true, y_score)
    mask = r >= target_recall
    if not mask.any(): return np.nan, np.nan
    idxs = np.where(mask)[0]
    best_i = int(idxs[np.argmax(p[idxs])])
    thr = float(t[max(best_i-1, 0)]) if len(t) else np.nan
    return float(p[best_i]), thr

def recall_at_precision(y_true, y_score, target_precision=0.90):
    p, r, t = precision_recall_curve(y_true, y_score)
    mask = p >= target_precision
    if not mask.any(): return np.nan, np.nan
    idxs = np.where(mask)[0]
    best_i = int(idxs[np.argmax(r[idxs])])
    thr = float(t[max(best_i-1, 0)]) if len(t) else np.nan
    return float(r[best_i]), thr

def calculate_all_metrics(all_preds: pd.DataFrame) -> Dict:
    metrics = {}
    for tool in ['SeqQC-Former','Mutect2','Strelka2']:
        if tool not in all_preds['TOOL'].unique(): continue
        df = all_preds[all_preds['TOOL'] == tool]
        y_true  = df['label'].astype(int).to_numpy()
        y_score = df['prob'].astype(float).to_numpy()
        pos = int((y_true == 1).sum()); neg = int((y_true == 0).sum())
        if pos == 0 or neg == 0:
            print(f"[WARN] {tool}: only one class present; skipping standard metrics.")
            metrics[tool] = {'AUROC': np.nan, 'AUPRC': np.nan, 'F1': np.nan,
                             'Precision': np.nan, 'Recall': np.nan, 'Specificity': np.nan,
                             'Prec@R0.95': np.nan, 'Rec@P0.90': np.nan,
                             'N': int(len(y_true)), 'Positive_Rate': float(y_true.mean())}
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        p, r, _ = precision_recall_curve(y_true, y_score)
        y_pred = (y_score >= 0.5).astype(int)
        p_at_r95, _ = precision_at_recall(y_true, y_score, 0.95)
        r_at_p90, _ = recall_at_precision(y_true, y_score, 0.90)

        metrics[tool] = {
            'AUROC': auc(fpr, tpr),
            'AUPRC': auc(r, p),
            'F1': f1_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'Specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            'Prec@R0.95': p_at_r95,
            'Rec@P0.90': r_at_p90,
            'N': int(len(y_true)),
            'Positive_Rate': float(y_true.mean())
        }
    return metrics

# ===================== Bootstrap utilities for CI =====================
def _roc_with_ci(y_true, y_score, n_boot=1000, seed=123):
    if len(np.unique(y_true)) < 2: return None
    rng = np.random.default_rng(seed)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    base_auc = auc(fpr, tpr)
    aucs = []
    idx = np.arange(len(y_true))
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        yt, ys = y_true[b], y_score[b]
        if len(np.unique(yt)) < 2: continue
        fb, tb, _ = roc_curve(yt, ys)
        aucs.append(auc(fb, tb))
    lo, hi = (base_auc, base_auc) if len(aucs)==0 else np.percentile(aucs, [2.5, 97.5])
    return {'fpr': fpr, 'tpr': tpr, 'auc': base_auc, 'auc_lo': float(lo), 'auc_hi': float(hi)}

def _pr_with_ci(y_true, y_score, n_boot=1000, seed=123):
    if len(np.unique(y_true)) < 2: return None
    rng = np.random.default_rng(seed)
    p, r, _ = precision_recall_curve(y_true, y_score)
    base_ap = auc(r, p)
    aps = []
    idx = np.arange(len(y_true))
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        yt, ys = y_true[b], y_score[b]
        if len(np.unique(yt)) < 2: continue
        pb, rb, _ = precision_recall_curve(yt, ys)
        aps.append(auc(rb, pb))
    lo, hi = (base_ap, base_ap) if len(aps)==0 else np.percentile(aps, [2.5, 97.5])
    return {'rec': r, 'prec': p, 'ap': base_ap, 'ap_lo': float(lo), 'ap_hi': float(hi)}

# ===================== Save helper (PNG only) =====================
def save_png(figpath: Path, dpi: int = 300):  # Reduced default DPI
    figpath = Path(figpath)
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(figpath) + '.png', bbox_inches='tight', dpi=dpi, 
                facecolor='white', edgecolor='none')  # Clean white background
    plt.close()

# ===================== UPDATED Plotting Functions =====================
def plot_roc(all_preds: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))  # Slightly smaller for publication
    plotted = False
    
    for tool in ['SeqQC-Former','Mutect2','Strelka2']:
        if tool not in all_preds['TOOL'].unique(): continue
        df = all_preds[all_preds['TOOL'] == tool]
        y = df['label'].astype(int).to_numpy()
        s = df['prob'].astype(float).to_numpy()
        stats = _roc_with_ci(y, s, n_boot=1000)
        if stats is None: continue
        
        color = TOOL_COLORS.get(tool)
        ci_pm = (stats['auc_hi'] - stats['auc_lo'])/2
        ax.plot(stats['fpr'], stats['tpr'], lw=LINEWIDTH,
                 label=f"{tool} (AUC {stats['auc']:.3f} ± {ci_pm:.3f})",
                 color=color)
        plotted = True
    
    # Diagonal reference line
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.7)
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Comparison', fontweight='bold', fontsize=12)
    
    # Clean styling
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.grid(False)  # Ensure no grid
    
    if plotted: 
        ax.legend(loc='lower right', frameon=True, fancybox=False, 
                 edgecolor='black', framealpha=1.0)
    
    save_png(outdir / 'ROC_Comparison', dpi=300)  # Reduced DPI

def plot_pr(all_preds: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    plotted = False
    
    # Store tool info for the legend box
    tool_info = []
    
    for tool in ['SeqQC-Former','Mutect2','Strelka2']:
        if tool not in all_preds['TOOL'].unique(): continue
        df = all_preds[all_preds['TOOL'] == tool]
        y = df['label'].astype(int).to_numpy()
        s = df['prob'].astype(float).to_numpy()
        stats = _pr_with_ci(y, s, n_boot=1000)
        if stats is None: continue
        
        color = TOOL_COLORS.get(tool)
        ci_pm = (stats['ap_hi'] - stats['ap_lo'])/2
        
        # Plot the curve
        ax.plot(stats['rec'], stats['prec'], lw=LINEWIDTH, color=color)
        plotted = True
        
        # Store tool info for the info box
        tool_info.append({
            'tool': tool,
            'ap': stats['ap'],
            'ci_pm': ci_pm,
            'color': color
        })
    
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision–Recall Comparison', fontweight='bold', fontsize=12)
    
    # Clean styling
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.grid(False)
    
    if plotted and tool_info:
        # Create a compact info box in bottom right corner
        info_text = "\n".join([f"{info['tool']}: AP {info['ap']:.3f} ± {info['ci_pm']:.3f}" 
                              for info in tool_info])
        
        # Add the info box to bottom right
        ax.text(0.98, 0.02, info_text, 
                transform=ax.transAxes,  # Use axis coordinates (0-1)
                fontsize=9,
                fontfamily=FONT_FAMILY,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", 
                         facecolor='white', 
                         edgecolor='black',
                         alpha=0.9,
                         linewidth=0.8),
                zorder=1000)  # Ensure it's on top
    
    save_png(outdir / 'PR_Curve', dpi=300)
def plot_score_distributions(all_preds: pd.DataFrame, outdir: Path):
    if all_preds.empty: return
    
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    order = [t for t in ['SeqQC-Former','Mutect2','Strelka2'] if t in all_preds['TOOL'].unique()]
    if not order: return
    
    # Use consistent colors
    palette = {0: CLASS_COLORS[0], 1: CLASS_COLORS[1]}
    
    # Create violin plot with clean styling
    sns.violinplot(
        data=all_preds, x='TOOL', y='prob', hue='label', split=True,
        palette=palette, inner=None, cut=0,
        order=order, linewidth=0.8, ax=ax
    )
    
    # Add boxplot outlines
    sns.boxplot(
        data=all_preds, x='TOOL', y='prob', hue='label',
        showcaps=False, boxprops={'facecolor':'none','edgecolor':'k','linewidth':1},
        whiskerprops={'linewidth':1}, showfliers=False, dodge=True, order=order, ax=ax
    )
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], ['Negative','Positive'], title='True Label', 
                 frameon=True, fancybox=False, edgecolor='black')
    
    ax.set_xlabel('', fontweight='bold')
    ax.set_ylabel('Prediction Score', fontweight='bold')
    ax.set_title('Score Distributions by True Class', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=0, ha='center')
    
    # Clean styling
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.grid(False)
    
    plt.tight_layout()
    save_png(outdir / 'Score_Distributions', dpi=300)  # Reduced DPI

def plot_confusion_matrices(all_preds: pd.DataFrame, outdir: Path):
    for tool in ['SeqQC-Former','Mutect2','Strelka2']:
        if tool not in all_preds['TOOL'].unique(): continue
        df = all_preds[all_preds['TOOL'] == tool]
        if len(df)==0 or len(np.unique(df['label']))<2: continue
        
        y   = df['label'].astype(int).to_numpy()
        yhat= (df['prob'].astype(float) >= 0.5).astype(int).to_numpy()
        cm_raw = confusion_matrix(y, yhat)
        cm_norm= confusion_matrix(y, yhat, normalize='true')
        
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        
        # Use colormap that matches tool color
        tool_color = TOOL_COLORS.get(tool, '#1f77b4')
        cmap = sns.light_palette(tool_color, as_cmap=True)
        
        sns.heatmap(cm_norm, annot=False, cmap=cmap, vmin=0, vmax=1, 
                   cbar=False, ax=ax,
                   xticklabels=['Pred 0','Pred 1'], 
                   yticklabels=['True 0','True 1'])
        
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.5, f"{cm_norm[i,j]*100:.1f}%\n({cm_raw[i,j]})",
                        ha='center', va='center', fontsize=11, 
                        fontweight='bold', color='black')
        
        ax.set_title(f'{tool} Confusion Matrix\n(Threshold=0.5)', 
                    fontweight='bold', fontsize=11)
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')
        
        plt.tight_layout()
        save_png(outdir / f'CM_{tool}', dpi=300)  # Reduced DPI

def plot_calibration(all_preds: pd.DataFrame, outdir: Path):
    for tool in ['SeqQC-Former','Mutect2','Strelka2']:
        if tool not in all_preds['TOOL'].unique(): continue
        df = all_preds[all_preds['TOOL'] == tool]
        y = df['label'].astype(int).to_numpy()
        p = df['prob'].astype(float).to_numpy()
        if len(np.unique(y)) < 2: continue
        
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy='quantile')
        bs = brier_score_loss(y, p)
        
        fig, ax = plt.subplots(figsize=(4.8, 4.3))
        
        # Use tool color
        color = TOOL_COLORS.get(tool)
        
        ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.7)
        ax.plot(mean_pred, frac_pos, marker='o', lw=LINEWIDTH, 
                markersize=MARKER_SIZE, color=color)
        
        ax.set_xlabel('Predicted Probability', fontweight='bold')
        ax.set_ylabel('Observed Fraction Positive', fontweight='bold')
        ax.set_title(f'Calibration: {tool}\n(Brier Score: {bs:.3f})', 
                    fontweight='bold', fontsize=11)
        
        # Clean styling
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        ax.grid(False)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        
        plt.tight_layout()
        save_png(outdir / f'Calibration_{tool}', dpi=300)  # Reduced DPI

# ---------- Overlap helpers ----------
def _compute_overlap_sets(all_preds: pd.DataFrame, thr: float = 0.5):
    sets = {}
    for tool in ['SeqQC-Former','Mutect2','Strelka2']:
        if tool in all_preds['TOOL'].unique():
            s = set(all_preds[(all_preds['TOOL']==tool) & (all_preds['prob']>=thr)]['key'])
            if len(s) > 0:
                sets[tool] = s
    return sets

def _overlap_counts_3(sets: Dict[str,set]) -> Dict[str,int]:
    A = sets.get('SeqQC-Former', set())
    B = sets.get('Mutect2', set())
    C = sets.get('Strelka2', set())
    return {
        'SeqQC_only': len(A - B - C),
        'Mutect2_only': len(B - A - C),
        'Strelka2_only': len(C - A - B),
        'SeqQC∩Mutect2_only': len((A & B) - C),
        'SeqQC∩Strelka2_only': len((A & C) - B),
        'Mutect2∩Strelka2_only': len((B & C) - A),
        'All_three': len(A & B & C),
        'Any': len(A | B | C)
    }

def _plot_overlap_bar(counts: Dict[str,int], outdir: Path, dpi: int = 300):  # Reduced DPI
    labels = [k for k in counts if k != 'Any']
    vals = [counts[k] for k in labels]
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    
    # Create colors based on tool involvement
    bar_colors = []
    for label in labels:
        if 'SeqQC' in label and 'Mutect2' not in label and 'Strelka2' not in label:
            bar_colors.append(TOOL_COLORS['SeqQC-Former'])
        elif 'Mutect2' in label and 'SeqQC' not in label and 'Strelka2' not in label:
            bar_colors.append(TOOL_COLORS['Mutect2'])
        elif 'Strelka2' in label and 'SeqQC' not in label and 'Mutect2' not in label:
            bar_colors.append(TOOL_COLORS['Strelka2'])
        elif 'All_three' in label:
            bar_colors.append('#7F7F7F')  # Gray for overlap
        else:
            bar_colors.append('#FFA500')  # Orange for partial overlaps
    
    bars = ax.bar(range(len(labels)), vals, color=bar_colors, edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Variant Count\n(Threshold ≥ 0.5)', fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('Variant Call Overlap Summary', fontweight='bold', fontsize=12)
    
    # Clean x-axis labels
    clean_labels = [l.replace('SeqQC', 'SeqQC-Former')
                   .replace('∩', ' ∩ ')
                   .replace('_only', ' Only')
                   .replace('_', ' ') for l in labels]
    
    ax.set_xticks(range(len(clean_labels)))
    ax.set_xticklabels(clean_labels, rotation=25, ha='right')
    
    # Clean styling
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.grid(False)
    
    plt.tight_layout()
    save_png(outdir / 'Overlap_Summary', dpi=dpi)

def plot_venn(all_preds: pd.DataFrame, outdir: Path):
    sets = _compute_overlap_sets(all_preds, thr=0.5)
    if len(sets) < 2:
        print("[INFO] Not enough non-empty sets for Venn; writing overlap summary only.")
        counts = _overlap_counts_3(sets)
        pd.DataFrame([counts]).to_csv(outdir / 'overlap_counts.csv', index=False)
        _plot_overlap_bar(counts, outdir, dpi=300)
        return

    total_any = len(set().union(*sets.values()))
    max_set = max(len(s) for s in sets.values())
    if total_any == 0 or max_set / max(total_any, 1) > 0.98:
        print("[INFO] Overlap extremely imbalanced; skipping Venn and writing summary instead.")
        counts = _overlap_counts_3(sets)
        pd.DataFrame([counts]).to_csv(outdir / 'overlap_counts.csv', index=False)
        _plot_overlap_bar(counts, outdir, dpi=300)
        return

    try:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        if len(sets) == 3 and HAS_VENN:
            v = venn3(list(sets.values()), set_labels=list(sets.keys()), ax=ax)
            # Apply consistent colors to Venn diagram
            if 'SeqQC-Former' in sets and v.get_label_by_id('100'):
                v.get_patch_by_id('100').set_color(TOOL_COLORS['SeqQC-Former'])
            if 'Mutect2' in sets and v.get_label_by_id('010'):
                v.get_patch_by_id('010').set_color(TOOL_COLORS['Mutect2'])
            if 'Strelka2' in sets and v.get_label_by_id('001'):
                v.get_patch_by_id('001').set_color(TOOL_COLORS['Strelka2'])
        elif len(sets) == 2 and HAS_VENN:
            v = venn2(list(sets.values()), set_labels=list(sets.keys()), ax=ax)
            # Apply consistent colors
            if 'SeqQC-Former' in sets and v.get_label_by_id('10'):
                v.get_patch_by_id('10').set_color(TOOL_COLORS['SeqQC-Former'])
            if 'Mutect2' in sets and v.get_label_by_id('01'):
                v.get_patch_by_id('01').set_color(TOOL_COLORS['Mutect2'])
        else:
            print("[INFO] matplotlib-venn not available; writing overlap summary instead.")
            counts = _overlap_counts_3(sets)
            pd.DataFrame([counts]).to_csv(outdir / 'overlap_counts.csv', index=False)
            _plot_overlap_bar(counts, outdir, dpi=300)
            return
        
        ax.set_title('Variant Call Overlap\n(Threshold = 0.5)', fontweight='bold', fontsize=12)
        save_png(outdir / 'Venn_All', dpi=300)  # Reduced DPI
    except MemoryError:
        print("[WARN] Venn plot ran out of memory; writing overlap summary instead.")
        plt.close()
        counts = _overlap_counts_3(sets)
        pd.DataFrame([counts]).to_csv(outdir / 'overlap_counts.csv', index=False)
        _plot_overlap_bar(counts, outdir, dpi=300)

# ===================== Paired bootstrap Δ (SeqQC - Caller) =====================
def _auc_pr(y_true, y_score):
    p, r, _ = precision_recall_curve(y_true, y_score)
    return auc(r, p)

def paired_bootstrap_delta(all_preds, tool_A="SeqQC-Former", tool_B="Mutect2", n_boot=1000, seed=123):
    dfA = all_preds[all_preds["TOOL"]==tool_A].sort_values("key")
    dfB = all_preds[all_preds["TOOL"]==tool_B].sort_values("key")
    if not np.array_equal(dfA["key"].to_numpy(), dfB["key"].to_numpy()):
        raise ValueError("Keys not aligned between tools (should not happen after truth alignment).")
    y = dfA["label"].astype(int).to_numpy()
    sA = dfA["prob"].astype(float).to_numpy()
    sB = dfB["prob"].astype(float).to_numpy()
    if len(np.unique(y)) < 2:
        return {"toolA": tool_A, "toolB": tool_B, "dAUROC": np.nan, "dAUROC_lo": np.nan, "dAUROC_hi": np.nan,
                "dAUPRC": np.nan, "dAUPRC_lo": np.nan, "dAUPRC_hi": np.nan}

    base_droc = roc_auc_score(y, sA) - roc_auc_score(y, sB)
    base_dpr  = _auc_pr(y, sA) - _auc_pr(y, sB)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    d_rocs, d_prs = [], []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        yb, a, bb = y[b], sA[b], sB[b]
        if len(np.unique(yb)) < 2:
            continue
        d_rocs.append(roc_auc_score(yb, a) - roc_auc_score(yb, bb))
        d_prs.append(_auc_pr(yb, a) - _auc_pr(yb, bb))
    def _ci(vals, base):
        if len(vals)==0: return (float(base), float(base))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)
    droc_lo, droc_hi = _ci(d_rocs, base_droc)
    dpr_lo,  dpr_hi  = _ci(d_prs,  base_dpr)
    return {"toolA": tool_A, "toolB": tool_B,
            "dAUROC": float(base_droc), "dAUROC_lo": droc_lo, "dAUROC_hi": droc_hi,
            "dAUPRC": float(base_dpr),  "dAUPRC_lo": dpr_lo,  "dAUPRC_hi": dpr_hi}

# ===================== Operating point table =====================
def _counts_at_threshold(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    TP = int(((y_true==1)&(y_pred==1)).sum())
    FP = int(((y_true==0)&(y_pred==1)).sum())
    TN = int(((y_true==0)&(y_pred==0)).sum())
    FN = int(((y_true==1)&(y_pred==0)).sum())
    return TP, FP, TN, FN

def _thr_for_recall(y_true, y_score, target_recall=0.95):
    p, r, t = precision_recall_curve(y_true, y_score)
    mask = r >= target_recall
    if not mask.any(): return np.nan
    idx = np.where(mask)[0].max()-1 if len(t) else 0
    return float(t[max(idx,0)]) if len(t) else np.nan

def _thr_for_precision(y_true, y_score, target_precision=0.98):
    p, r, t = precision_recall_curve(y_true, y_score)
    mask = p >= target_precision
    if not mask.any(): return np.nan
    idxs = np.where(mask)[0]
    best_i = int(idxs[np.argmax(r[idxs])])
    return float(t[max(best_i-1,0)]) if len(t) else np.nan

def make_operating_point_table(all_preds, outdir):
    rows = []
    for tool in ['SeqQC-Former','Mutect2','Strelka2']:
        df = all_preds[all_preds['TOOL']==tool]
        y = df['label'].astype(int).to_numpy()
        s = df['prob'].astype(float).to_numpy()

        for spec in [("Recall@0.95", _thr_for_recall(y,s,0.95)),
                     ("Precision@0.98", _thr_for_precision(y,s,0.98)),
                     ("Precision@0.90", _thr_for_precision(y,s,0.90))]:
            name, thr = spec
            if np.isnan(thr):
                rows.append({"Tool":tool,"Setting":name,"Threshold":np.nan,"Precision":np.nan,"Recall":np.nan,
                             "TP":np.nan,"FP":np.nan,"TN":np.nan,"FN":np.nan})
            else:
                TP,FP,TN,FN = _counts_at_threshold(y,s,thr)
                prec = TP/(TP+FP) if (TP+FP)>0 else np.nan
                rec  = TP/(TP+FN) if (TP+FN)>0 else np.nan
                rows.append({"Tool":tool,"Setting":name,"Threshold":thr,"Precision":prec,"Recall":rec,
                             "TP":TP,"FP":FP,"TN":TN,"FN":FN})
    op = pd.DataFrame(rows)
    op.to_csv(outdir / "operating_points.csv", index=False, float_format="%.6f")
    return op

# ===================== Stratified PR by VAF / Depth =====================
def _maybe_annotations_for_bins(seqqc_raw, mutect_raw, strelka_raw):
    frames = []
    for df in [seqqc_raw.assign(SOURCE="seqqc"), mutect_raw.assign(SOURCE="mutect"), strelka_raw.assign(SOURCE="strelka")]:
        d = df.copy()
        if "key" not in d.columns:
            try: d = ensure_key(d)
            except Exception: continue
        low = {c.lower(): c for c in d.columns}
        def has(c): return c.lower() in low
        def col(c): return low[c.lower()]

        vaf = None; depth = None
        for af_name in ["AF","af","TumorAF","tumor_af","FA","fa"]:
            if has(af_name):
                vaf = pd.to_numeric(d[col(af_name)], errors="coerce"); break
        if vaf is None:
            ref_cands = [n for n in low if n in {"t_ref_count","tumor_ref_count","tref","ref_count"}]
            alt_cands = [n for n in low if n in {"t_alt_count","tumor_alt_count","talt","alt_count"}]
            if ref_cands and alt_cands:
                tref = pd.to_numeric(d[low[ref_cands[0]]], errors="coerce").clip(lower=0)
                talt = pd.to_numeric(d[low[alt_cands[0]]], errors="coerce").clip(lower=0)
                denom = (tref + talt).replace(0, np.nan)
                vaf = (talt / denom).astype(float)
                depth = (tref + talt).astype(float)
        if vaf is None:
            for ad_name in ["AD","ad","TUMOR_AD","tumor_ad"]:
                if ad_name.lower() in low:
                    parts = d[col(ad_name)].astype(str).str.split("[,:;]", regex=True)
                    try:
                        t_ref = pd.to_numeric(parts.str[0], errors="coerce").clip(lower=0)
                        t_alt = pd.to_numeric(parts.str[1], errors="coerce").clip(lower=0)
                        denom = (t_ref + t_alt).replace(0, np.nan)
                        vaf = (t_alt / denom).astype(float)
                        depth = (t_ref + t_alt).astype(float)
                        break
                    except Exception:
                        pass
        if vaf is not None:
            out = d[["key"]].copy()
            out["vaf_est"] = vaf.clip(0,1)
            out["depth_est"] = depth if depth is not None else np.nan
            frames.append(out)
    if not frames:
        return None
    ann = pd.concat(frames, ignore_index=True).drop_duplicates("key")
    return ann

def stratified_pr_plots(all_preds, outdir, seqqc_raw, mutect_raw, strelka_raw):
    ann = _maybe_annotations_for_bins(seqqc_raw, mutect_raw, strelka_raw)
    if ann is None or ann.empty:
        print("[INFO] No AF/depth columns found; skipping stratified PR.")
        return
    df = all_preds.merge(ann, on="key", how="left")
    # VAF bins
    bins = [0.0, 0.05, 0.10, 0.20, 1.01]
    labels = ["<5%", "5–10%", "10–20%", "≥20%"]
    df["VAF_bin"] = pd.cut(df["vaf_est"], bins=bins, labels=labels, right=False, include_lowest=True)
    # Depth tertiles
    non_na_depth = df["depth_est"].dropna()
    if non_na_depth.empty:
        depth_edges = None
    else:
        qs = non_na_depth.quantile([0, 1/3, 2/3, 1]).to_list()
        for i in range(1, len(qs)):
            if qs[i] <= qs[i-1]: qs[i] = qs[i-1] + 1e-9
        depth_edges = qs
        df["Depth_bin"] = pd.cut(df["depth_est"], bins=depth_edges, labels=["low","med","high"], include_lowest=True)

    # PR by VAF
    for b in labels:
        sub = df[df["VAF_bin"] == b]
        if sub["label"].notna().sum() == 0: continue
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        plotted=False
        for tool in ['SeqQC-Former','Mutect2','Strelka2']:
            dft = sub[sub["TOOL"]==tool]
            if dft["label"].nunique()<2: continue
            p,r,_ = precision_recall_curve(dft["label"].astype(int), dft["prob"].astype(float))
            ap = auc(r,p)
            ax.plot(r,p,lw=LINEWIDTH,label=f"{tool} (AP {ap:.3f})", color=TOOL_COLORS.get(tool))
            plotted=True
        if not plotted:
            plt.close(); continue
        ax.set_xlim([0,1]); ax.set_ylim([0,1])
        ax.set_xlabel("Recall", fontweight='bold')
        ax.set_ylabel("Precision", fontweight='bold')
        ax.set_title(f"PR by VAF Bin: {b}", fontweight='bold', fontsize=12)
        ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor='black')
        ax.grid(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        save_png(outdir / f"PR_byVAF_{b.replace('%','pct').replace('–','-')}", dpi=300)  # Reduced DPI

    # PR by Depth
    if depth_edges is not None:
        for b in ["low","med","high"]:
            sub = df[df["Depth_bin"] == b]
            if sub["label"].notna().sum() == 0: continue
            fig, ax = plt.subplots(figsize=(5.5, 5.0))
            plotted=False
            for tool in ['SeqQC-Former','Mutect2','Strelka2']:
                dft = sub[sub["TOOL"]==tool]
                if dft["label"].nunique()<2: continue
                p,r,_ = precision_recall_curve(dft["label"].astype(int), dft["prob"].astype(float))
                ap = auc(r,p)
                ax.plot(r,p,lw=LINEWIDTH,label=f"{tool} (AP {ap:.3f})", color=TOOL_COLORS.get(tool))
                plotted=True
            if not plotted:
                plt.close(); continue
            ax.set_xlim([0,1]); ax.set_ylim([0,1])
            ax.set_xlabel("Recall", fontweight='bold')
            ax.set_ylabel("Precision", fontweight='bold')
            ax.set_title(f"PR by Depth Bin: {b}", fontweight='bold', fontsize=12)
            ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor='black')
            ax.grid(False)
            ax.spines['bottom'].set_linewidth(0.8)
            ax.spines['left'].set_linewidth(0.8)
            save_png(outdir / f"PR_byDepth_{b}", dpi=300)  # Reduced DPI

# ===================== Error analysis: top FP/FN =====================
def dump_top_errors(all_preds, outdir, topn=20):
    base = all_preds.copy()
    base[["chrom","pos","ref","alt"]] = base["key"].str.split(":", expand=True)
    base["pos"] = pd.to_numeric(base["pos"], errors="coerce")
    rows = []
    for tool in ['SeqQC-Former','Mutect2','Strelka2']:
        df = base[base["TOOL"]==tool].copy()
        fp = df[df["label"]==0].sort_values("prob", ascending=False).head(topn); fp["ERR"]="FP"
        fn = df[df["label"]==1].sort_values("prob", ascending=True).head(topn);  fn["ERR"]="FN"
        rows.append(pd.concat([fp,fn], ignore_index=True))
    out = pd.concat(rows, ignore_index=True)
    cols = ["TOOL","ERR","key","chrom","pos","ref","alt","prob","label"]
    out[cols].to_csv(outdir / "top_errors_by_tool.csv", index=False)
    with open(outdir / "igv_top_errors.batch", "w", encoding="utf-8") as fh:
        fh.write("# IGV loci list for top FPs/FNs\n")
        for _, r in out.iterrows():
            if pd.isna(r["pos"]): continue
            fh.write(f"{r['chrom']}:{int(r['pos'])}-{int(r['pos'])}\n")

# ===================== Data Load & Merge =====================
def load_data(data_root: Path, seqqc_path: Path, mutect_path: Path, strelka_path: Path) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = load_truth_df(data_root)

    # SeqQC
    seqqc_raw = pd.read_csv(seqqc_path)
    if 'key' not in seqqc_raw.columns: seqqc_raw = ensure_key(seqqc_raw)
    if 'prob' not in seqqc_raw.columns: seqqc_raw = ensure_prob(seqqc_raw, "SeqQC-Former")
    seqqc = align_to_truth_universe(truth, seqqc_raw, "SeqQC-Former")

    # Mutect2
    mutect_raw = pd.read_csv(mutect_path)
    if 'key' not in mutect_raw.columns: mutect_raw = ensure_key(mutect_raw)
    mutect_raw = ensure_prob(mutect_raw, "Mutect2")
    mutect = align_to_truth_universe(truth, mutect_raw, "Mutect2")

    # Strelka2
    strelka_raw = pd.read_csv(strelka_path)
    if 'key' not in strelka_raw.columns: strelka_raw = ensure_key(strelka_raw)
    strelka_raw = ensure_prob(strelka_raw, "Strelka2")
    strelka = align_to_truth_universe(truth, strelka_raw, "Strelka2")

    all_preds = pd.concat([seqqc, mutect, strelka], ignore_index=True)

    for tool in ['SeqQC-Former','Mutect2','Strelka2']:
        df = all_preds[all_preds['TOOL'] == tool]
        _class_counts(df, tool)

    if all_preds.empty:
        raise ValueError("No predictions after merging; check your inputs.")
    return all_preds, calculate_all_metrics(all_preds), seqqc_raw, mutect_raw, strelka_raw

# ===================== Main =====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare SeqQC-Former with Mutect2 and Strelka2 (paired, PNG @ 300 dpi)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config',   type=Path, default=None)
    parser.add_argument('--data_root',type=Path, default=None)
    parser.add_argument('--seqqc',    type=Path, default=None)
    parser.add_argument('--mutect',   type=Path, default=None)
    parser.add_argument('--strelka',  type=Path, default=None)
    parser.add_argument('--outdir',   type=Path, default=None)
    args = parser.parse_args()
    args = resolve_paths(args)
    args.outdir.mkdir(exist_ok=True, parents=True)

    print("\n[1/6] Loading data...")
    if args.config: print(f"Config: {args.config}")
    print(f"Data root: {args.data_root}")
    print(f"SeqQC file: {args.seqqc}")
    print(f"Mutect2 file: {args.mutect}")
    print(f"Strelka2 file: {args.strelka}")
    print(f"Outdir: {args.outdir}")

    all_preds, metrics, seqqc_raw, mutect_raw, strelka_raw = load_data(args.data_root, args.seqqc, args.mutect, args.strelka)

    print("\n[2/6] Generating visualizations:")
    for name, func in [
        ("ROC Curve", plot_roc),
        ("PR Curve",  plot_pr),
        ("Score Distributions", plot_score_distributions),
        ("Confusion Matrices",  plot_confusion_matrices),
        ("Calibration Curves",  plot_calibration),
        ("Venn / Overlap",      plot_venn),
    ]:
        print(f"- {name}")
        try:
            func(all_preds, args.outdir)
        except MemoryError as e:
            print(f"  [WARN] Memory error in {name}: {e}. Skipping this plot.")
            plt.close('all')  # Clear any remaining figures
        except Exception as e:
            print(f"  [ERROR] Failed to generate {name}: {e}")
            plt.close('all')

    print("\n[3/6] Saving performance metrics...")
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.to_csv(args.outdir / 'performance_metrics.csv', float_format='%.4f')

    # 3b) Paired bootstrap deltas
    print("\n[3b/6] Paired bootstrap Δ metrics...")
    d1 = paired_bootstrap_delta(all_preds, "SeqQC-Former", "Mutect2", n_boot=1000)
    d2 = paired_bootstrap_delta(all_preds, "SeqQC-Former", "Strelka2", n_boot=1000)
    pd.DataFrame([d1,d2]).to_csv(args.outdir / "delta_metrics_bootstrap.csv", index=False)
    print(pd.DataFrame([d1,d2]))

    # 3c) Operating points
    print("\n[3c/6] Operating-point table...")
    op_tbl = make_operating_point_table(all_preds, args.outdir)
    print(op_tbl)

    # 3d) Stratified PR
    print("\n[3d/6] Stratified PR by VAF/depth (if available)...")
    try:
        stratified_pr_plots(all_preds, args.outdir, seqqc_raw, mutect_raw, strelka_raw)
    except MemoryError as e:
        print(f"  [WARN] Memory error in stratified PR: {e}. Skipping.")
    except Exception as e:
        print(f"  [ERROR] Failed stratified PR: {e}")

    # 3e) Top errors
    print("\n[3e/6] Dumping top errors (FP/FN) and IGV batch...")
    dump_top_errors(all_preds, args.outdir, topn=20)

    print("\n[4/6] Generating LaTeX table (no Jinja2)...")
    cols = ['AUROC','AUPRC','F1','Precision','Recall','Specificity','Prec@R0.95','Rec@P0.90']
    header = " & ".join(["Tool"] + cols) + " \\\\ \\hline\n"
    body = ""
    for tool, row in metrics_df.fillna("").iterrows():
        vals = []
        for c in cols:
            v = row.get(c, "")
            vals.append(f"{v:.3f}" if isinstance(v,(int,float,np.floating)) and v==v else "")
        body += f"{tool} & " + " & ".join(vals) + " \\\\\n"
    latex_table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\begin{tabular}{lrrrrrrrr}\n" + header + body +
        "\\end{tabular}\n"
        "\\caption{Performance comparison on the same labeled universe.}\n"
        "\\label{tab:performance}\n\\end{table}\n"
    )
    with open(args.outdir / 'performance_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)

    print("\n[5/6] Generating markdown report...")
    header_md = "| Tool | " + " | ".join(cols) + " |\n"
    sep_md    = "|" + "|".join(["---"]*(len(cols)+1)) + "|\n"
    lines = []
    for tool, row in metrics_df.iterrows():
        vals = []
        for c in cols:
            v = row.get(c)
            vals.append("" if (not isinstance(v,(int,float,np.floating)) or v!=v) else f"{v:.3f}")
        lines.append(f"| {tool} | " + " | ".join(vals) + " |\n")
    md_table = header_md + sep_md + "".join(lines)
    with open(args.outdir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(f"""# SeqQC-Former Comparison Results

## Performance Metrics
{md_table}

## Generated Figures (PNG @ 300 dpi)
- `ROC_Comparison.png`
- `PR_Curve.png`
- `Score_Distributions.png`
- `Calibration_*.png`
- `CM_*.png`
- `Venn_All.png` (skipped if unstable; see `Overlap_Summary.png` instead)
- `Overlap_Summary.png` (written when Venn is skipped)
- `overlap_counts.csv`

## Extra Analyses
- `delta_metrics_bootstrap.csv` (paired ΔAUROC/ΔAUPRC with 95% CIs)
- `operating_points.csv` (Recall@0.95, Precision@0.98 & 0.90 with TP/FP/TN/FN)
- `PR_byVAF_*.png` and `PR_byDepth_*.png` (if AF/depth available)
- `top_errors_by_tool.csv` and `igv_top_errors.batch`
""")

    print("\n[6/6] Results summary:")
    print(f"\n{'Tool':<15} {'AUROC':<8} {'AUPRC':<8} {'F1':<6} {'Precision':<9} {'Recall':<7} {'Specificity':<10} {'Prec@R0.95':<11} {'Rec@P0.90':<10}")
    print("-" * 95)
    for tool, vals in metrics.items():
        def fmt(x):
            return "nan" if (x is None or (isinstance(x,float) and x!=x)) else f"{x:.3f}"
        print(f"{tool:<15} {fmt(vals['AUROC'])}   {fmt(vals['AUPRC'])}   {fmt(vals['F1'])}   {fmt(vals['Precision'])}      {fmt(vals['Recall'])}      {fmt(vals['Specificity'])}      {fmt(vals['Prec@R0.95']):<11} {fmt(vals['Rec@P0.90']):<10}")

    print(f"\nResults saved to: {args.outdir}")
    print("Generated files:")
    for fpath in sorted(args.outdir.glob('*')):
        print(f"  - {fpath.name}")