"""
AP 실험 — 불균형 대응 기법 비교 (ap_experiment.py)
=====================================================
운영 주 레이블 label_entry(양성 10.6%)는 클래스 불균형이 크다.
랭킹 품질 지표 Average Precision(AP)을 기준으로,
재조정 기법 4종이 실제로 AP를 개선하는지 공정하게 측정한다.

비교 조건 (6개, 모두 동일 RepeatedStratifiedKFold 분할 공유 → paired 비교):
  GB_baseline       : GB, 재조정 없음 (기준점)
  GB_sample_weight  : GB + compute_sample_weight("balanced") — train fold에서만 계산
                      ※ GB엔 class_weight 인자가 없어, balanced 가중을 sample_weight로 동치 구현
  GB_SMOTE          : GB + SMOTE(train fold만 오버샘플)  ※ imblearn 없으면 자동 skip
  GB_HPtuned        : GB 하이퍼파라미터 nested CV 튜닝 (scoring=average_precision)
  Ensemble          : LR(balanced) + GB(sample_weight) predict_proba 수동 평균
  LR_baseline       : LR class_weight=balanced (참조선)

지표 (모두 out-of-fold 확률 랭킹으로 계산, 임계값 무관):
  AP            : Average Precision (PR곡선 아래 면적). 주지표
  AUC           : ROC 곡선 아래 면적 (위험>안전 순위 일치 확률). 종합 랭킹 품질
  P@top-N       : 위험 상위 N개 동을 뽑을 때 실제 양성 비율 (운영 "선제 배치" 시나리오)
  R-precision   : 상위 (양성 수)개의 precision. 레이블 난이도에 자동 적응, AP의 짝

검증:
  RepeatedStratifiedKFold — repeat당 OOF 지표 1개, repeat 간 mean±std.
  N_REPEATS=10·20 둘 다 실행해 추정 안정성 비교 후 채택.

누출 차단: 대치/스케일/가중/SMOTE/HP탐색 전부 train fold에서만 적합.

산출물 (Outputs/ap_experiment/):
  ap_summary.md             — 두 레이블 통합 비교 리포트 (10 vs 20 안정성 포함)
  ap_raw_{label}_r{n}.csv   — 조건 × repeat 별 지표 원자료
  ap_boxplot_{label}.png    — 조건별 AP 분포 박스플롯 (주력 20repeat)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

# imblearn은 선택적 의존성 — 없으면 SMOTE 조건만 건너뜀
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

# ── 폰트 (train_transition_model.py 패턴 그대로) ────────────────────────
_FONT_FILE = "C:/Windows/Fonts/malgun.ttf"
if Path(_FONT_FILE).exists():
    fp = fm.FontProperties(fname=_FONT_FILE)
    plt.rcParams["font.family"] = fp.get_name()
else:
    plt.rcParams["font.family"] = "Malgun Gothic"
    fp = fm.FontProperties()
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 / 상수 ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "Outputs" / "전이예측" / "transition_features.csv"
OUT  = ROOT / "Outputs" / "ap_experiment"
OUT.mkdir(parents=True, exist_ok=True)

FEAT_PREFIXES = ["level_", "delta_", "infra_"]   # save_model.py와 동일 (dep_* 누출 제외)
LABELS        = ["label_entry", "label_slope"]
RANDOM_STATE  = 42

N_SPLITS       = 5
N_REPEATS_LIST = [10, 20]   # 둘 다 실행해 추정 안정성 비교. 마지막(20)이 주력
PRIMARY_REPEAT = 20         # 박스플롯·주력 표에 쓸 repeat 수
TOP_N          = 20         # P@top-N의 N — "위험 상위 20개 동 선제 배치" 운영 시나리오
HP_N_ITER      = 15         # RandomizedSearchCV 시도 수
HP_CV          = 3          # nested inner CV fold 수

# 운영 모델 파라미터 (save_model.py:68-72)
GB_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                 min_samples_leaf=10, subsample=0.8, random_state=RANDOM_STATE)
LR_PARAMS = dict(C=0.1, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)

# HP 탐색공간 (N 작음 → 과적합 억제 위주)
HP_GRID = {
    "clf__n_estimators":     [50, 100, 200],
    "clf__max_depth":        [2, 3],
    "clf__learning_rate":    [0.03, 0.05, 0.1],
    "clf__min_samples_leaf": [5, 10, 20],
    "clf__subsample":        [0.7, 0.8, 1.0],
}


# ── 파이프라인 빌더 ───────────────────────────────────────────────────────
def gb_pipe(**overrides):
    params = {**GB_PARAMS, **overrides}
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(**params)),
    ])

def lr_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(**LR_PARAMS)),
    ])


# ── 조건별 fold 학습기: (X_tr, y_tr, X_te) → test fold 양성 확률 ───────────
def fit_gb_baseline(X_tr, y_tr, X_te):
    p = gb_pipe(); p.fit(X_tr, y_tr)
    return p.predict_proba(X_te)[:, 1]

def fit_gb_sample_weight(X_tr, y_tr, X_te):
    sw = compute_sample_weight("balanced", y_tr)   # train fold에서만 계산 (누출 차단)
    p = gb_pipe(); p.fit(X_tr, y_tr, clf__sample_weight=sw)
    return p.predict_proba(X_te)[:, 1]

def fit_gb_smote(X_tr, y_tr, X_te):
    imp = SimpleImputer(strategy="median")         # SMOTE는 결측 불가 → 대치 후 적용
    X_tr_i = imp.fit_transform(X_tr)
    X_te_i = imp.transform(X_te)
    k = min(5, int(y_tr.sum()) - 1)                # k_neighbors < 소수클래스 표본수
    if k < 1:
        return fit_gb_sample_weight(X_tr, y_tr, X_te)   # 양성 1개뿐이면 가중으로 폴백
    X_rs, y_rs = SMOTE(k_neighbors=k, random_state=RANDOM_STATE).fit_resample(X_tr_i, y_tr)
    clf = GradientBoostingClassifier(**GB_PARAMS)
    clf.fit(X_rs, y_rs)
    return clf.predict_proba(X_te_i)[:, 1]

def fit_gb_hptuned(X_tr, y_tr, X_te):
    inner = StratifiedKFold(n_splits=HP_CV, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        gb_pipe(), HP_GRID, n_iter=HP_N_ITER, scoring="average_precision",
        cv=inner, random_state=RANDOM_STATE, n_jobs=-1, error_score="raise",
    )
    search.fit(X_tr, y_tr)                          # nested: outer train 내부에서만 탐색
    return search.best_estimator_.predict_proba(X_te)[:, 1]

def fit_ensemble(X_tr, y_tr, X_te):
    sw = compute_sample_weight("balanced", y_tr)
    gb = gb_pipe(); gb.fit(X_tr, y_tr, clf__sample_weight=sw)
    lr = lr_pipe(); lr.fit(X_tr, y_tr)
    return (gb.predict_proba(X_te)[:, 1] + lr.predict_proba(X_te)[:, 1]) / 2.0  # 소프트 평균

def fit_lr_baseline(X_tr, y_tr, X_te):
    p = lr_pipe(); p.fit(X_tr, y_tr)
    return p.predict_proba(X_te)[:, 1]

CONDITIONS = [
    ("GB_baseline",      fit_gb_baseline),
    ("GB_sample_weight", fit_gb_sample_weight),
    ("GB_SMOTE",         fit_gb_smote),
    ("GB_HPtuned",       fit_gb_hptuned),
    ("Ensemble",         fit_ensemble),
    ("LR_baseline",      fit_lr_baseline),
]

METRICS = ["ap", "auc", "p_at_n", "r_precision"]


# ── 지표 계산 (OOF 확률 랭킹) ─────────────────────────────────────────────
def precision_at_k(y_true, scores, k):
    k = min(int(k), len(y_true))
    if k < 1:
        return np.nan
    top = np.argsort(scores)[::-1][:k]             # 확률 내림차순 상위 k
    return float(y_true[top].mean())

def eval_oof(y, oof, n_pos):
    return {
        "ap":          average_precision_score(y, oof),
        "auc":         roc_auc_score(y, oof),
        "p_at_n":      precision_at_k(y, oof, TOP_N),
        "r_precision": precision_at_k(y, oof, n_pos),   # k = 양성 수
    }


# ── 한 (레이블, n_repeats) 실험 ───────────────────────────────────────────
def run_label(X, y, label, n_repeats, chance, save_plot=False):
    n, npos = len(y), int(y.sum())
    conditions = [c for c in CONDITIONS if not (c[0] == "GB_SMOTE" and not HAS_IMBLEARN)]

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats,
                                   random_state=RANDOM_STATE)
    splits = list(rskf.split(X, y))
    repeats = [splits[i * N_SPLITS:(i + 1) * N_SPLITS] for i in range(n_repeats)]

    # mat[metric][condition] = [repeat별 값 ...]
    mat = {m: {name: [] for name, _ in conditions} for m in METRICS}
    for r, folds in enumerate(repeats):
        for name, fit_fn in conditions:
            oof = np.full(n, np.nan)
            for tr, te in folds:
                oof[te] = fit_fn(X[tr], y[tr], X[te])
            for m, v in eval_oof(y, oof, npos).items():
                mat[m][name].append(v)
        print(f"    repeat {r+1:2d}/{n_repeats} 완료")

    # 요약 + AP에 대한 GB_baseline 대비 paired 검정
    base_ap = np.array(mat["ap"]["GB_baseline"])
    rows = []
    for name, _ in conditions:
        ap = np.array(mat["ap"][name])
        if name == "GB_baseline":
            pval, winrate = np.nan, np.nan
        else:
            diff = ap - base_ap
            pval = wilcoxon(ap, base_ap).pvalue if np.any(diff != 0) else 1.0
            winrate = float(np.mean(diff > 0))
        auc = np.array(mat["auc"][name])
        rows.append({
            "label": label, "n_repeats": n_repeats, "condition": name,
            "ap_mean": ap.mean(), "ap_std": ap.std(ddof=1),
            "ap_lift_vs_chance": ap.mean() - chance,
            "ap_lift_vs_baseline": ap.mean() - base_ap.mean(),
            "ap_wilcoxon_p": pval, "ap_winrate": winrate,
            "auc_mean": auc.mean(), "auc_std": auc.std(ddof=1),
            "p_at_n_mean": np.mean(mat["p_at_n"][name]),
            "p_at_n_std":  np.std(mat["p_at_n"][name], ddof=1),
            "rprec_mean":  np.mean(mat["r_precision"][name]),
            "rprec_std":   np.std(mat["r_precision"][name], ddof=1),
        })
        tag = "  (기준)" if name == "GB_baseline" else ""
        print(f"    [{name:17s}] AP={ap.mean():.3f}±{ap.std(ddof=1):.3f}  "
              f"AUC={auc.mean():.3f}  P@{TOP_N}={np.mean(mat['p_at_n'][name]):.3f}  "
              f"Rprec={np.mean(mat['r_precision'][name]):.3f}{tag}")

    summary = pd.DataFrame(rows)

    # raw 저장 (wide: 지표 3종)
    names = [nm for nm, _ in conditions]
    raw = pd.DataFrame({
        "label": label, "n_repeats": n_repeats,
        "repeat":    np.tile(np.arange(1, n_repeats + 1), len(names)),
        "condition": np.repeat(names, n_repeats),
        "ap":        np.concatenate([mat["ap"][nm]          for nm in names]),
        "auc":       np.concatenate([mat["auc"][nm]         for nm in names]),
        "p_at_n":    np.concatenate([mat["p_at_n"][nm]      for nm in names]),
        "r_precision": np.concatenate([mat["r_precision"][nm] for nm in names]),
    })
    raw.to_csv(OUT / f"ap_raw_{label}_r{n_repeats}.csv", index=False, encoding="utf-8-sig")

    # 박스플롯 (주력 repeat에서만)
    if save_plot:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.boxplot([mat["ap"][nm] for nm in names], labels=names, showmeans=True)
        ax.axhline(chance, color="#e74c3c", ls="--", lw=1.2, label=f"chance AP = {chance:.3f}")
        ax.set_ylabel("Average Precision", fontproperties=fp)
        ax.set_title(f"AP 분포 비교 — {label}  (N={n}, 양성 {chance:.1%}, "
                     f"{n_repeats}회 반복 5-fold)", fontproperties=fp, fontsize=12)
        ax.set_xticklabels(names, rotation=20, ha="right", fontproperties=fp, fontsize=9)
        ax.legend(prop=fp, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT / f"ap_boxplot_{label}.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"    저장: ap_boxplot_{label}.png")

    print(f"    저장: ap_raw_{label}_r{n_repeats}.csv")
    return summary


# ── 리포트 작성 ───────────────────────────────────────────────────────────
def write_report(all_summary: pd.DataFrame, chances: dict):
    def fmt_p(p):
        if pd.isna(p):
            return "—"
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    lines = [
        "# AP 실험 — 불균형 대응 기법 비교",
        "",
        f"- 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 검증: RepeatedStratifiedKFold (n_splits={N_SPLITS}, n_repeats={'·'.join(map(str, N_REPEATS_LIST))}) — "
        "repeat당 out-of-fold 지표 1개, repeat 간 mean±std",
        f"- 주력: **n_repeats={PRIMARY_REPEAT}** (아래 기법 비교표). 10 vs 20 안정성은 부록 표 참고",
        "- 지표: **AP**(임계값 불변 랭킹 지표, 주지표) · "
        "**AUC**(위험>안전 순위 일치 확률, 종합 랭킹 품질) · "
        f"**P@{TOP_N}**(위험 상위 {TOP_N}개 동의 실제 양성 비율) · "
        "**R-precision**(상위 양성수개의 precision, 레이블 난이도에 자동 적응)",
        "- GB는 class_weight 인자가 없어 `compute_sample_weight(\"balanced\")`로 동치 구현. "
        "모든 재조정/대치/탐색은 train fold에서만 적합(누출 차단)",
    ]
    if not HAS_IMBLEARN:
        lines.append("- ⚠️ imbalanced-learn 미설치 → GB_SMOTE 조건 제외됨")
    lines.append("")

    for label in LABELS:
        ch = chances[label]
        prim = all_summary[(all_summary["label"] == label) &
                           (all_summary["n_repeats"] == PRIMARY_REPEAT)]
        if prim.empty:
            continue
        best = prim.loc[prim["ap_mean"].idxmax(), "condition"]

        lines += [
            f"## {label}  (chance AP = {ch:.3f}, P@{TOP_N}·R-prec의 무작위 기대값 = {ch:.3f})",
            "",
            f"### 기법 비교 (n_repeats={PRIMARY_REPEAT})",
            "",
            f"| 조건 | AP | AUC | vs chance | vs 기준 | Wilcoxon p | 승률 | P@{TOP_N} | R-prec |",
            "|------|-----|-----|-----------|---------|------------|------|--------|--------|",
        ]
        for _, r in prim.iterrows():
            star = " ⭐" if r["condition"] == best else ""
            wr = "—" if pd.isna(r["ap_winrate"]) else f"{r['ap_winrate']:.0%}"
            lines.append(
                f"| {r['condition']}{star} | {r['ap_mean']:.3f}±{r['ap_std']:.3f} | "
                f"{r['auc_mean']:.3f}±{r['auc_std']:.3f} | "
                f"{r['ap_lift_vs_chance']:+.3f} | {r['ap_lift_vs_baseline']:+.3f} | "
                f"{fmt_p(r['ap_wilcoxon_p'])} | {wr} | "
                f"{r['p_at_n_mean']:.3f}±{r['p_at_n_std']:.3f} | "
                f"{r['rprec_mean']:.3f}±{r['rprec_std']:.3f} |"
            )

        # 10 vs 20 안정성
        lines += [
            "",
            "### 추정 안정성: 10 vs 20 repeat (AP mean±std)",
            "",
            "| 조건 | n=10 | n=20 | std 변화 |",
            "|------|------|------|----------|",
        ]
        for cond in prim["condition"]:
            r10 = all_summary[(all_summary["label"] == label) &
                              (all_summary["n_repeats"] == 10) &
                              (all_summary["condition"] == cond)]
            r20 = all_summary[(all_summary["label"] == label) &
                              (all_summary["n_repeats"] == 20) &
                              (all_summary["condition"] == cond)]
            if r10.empty or r20.empty:
                continue
            s10, s20 = r10.iloc[0], r20.iloc[0]
            arrow = "↓ 안정" if s20["ap_std"] < s10["ap_std"] else "↑"
            lines.append(
                f"| {cond} | {s10['ap_mean']:.3f}±{s10['ap_std']:.3f} | "
                f"{s20['ap_mean']:.3f}±{s20['ap_std']:.3f} | {arrow} |"
            )
        lines += ["", f"> ⭐ = n_repeats={PRIMARY_REPEAT} 최고 평균 AP. 승률 = GB_baseline 대비 repeat별 AP 우세 비율.", ""]

    lines += [
        "## 해석 가이드",
        "",
        "- **AP / P@N / R-prec 모두 chance를 넘어야** 예측력 있음. 세 지표가 같은 방향이면 결론이 견고함을 의미.",
        f"- **P@{TOP_N}**: 가장 직관적. \"위험 상위 {TOP_N}개 동을 선제 배치하면 그중 몇 %가 진짜 위험이냐\"를 의미.",
        "- **R-precision**: 레이블마다 양성 수가 달라(P@20이 천장을 칠 수 있음) 공정 비교용. AP와 거의 같이 움직임.",
        "- **vs 기준 + Wilcoxon p<0.05 + 승률↑**: 그 기법이 GB_baseline 대비 통계적으로 유의한 개선을 의미.",
        "- ⚠️ **std·p값은 \"같은 데이터의 분할 안정성\"이지 \"새 데이터 일반화 보장\"이 아님.** "
        "N이 작아(359/419) 실제 일반화 오차는 이보다 큼 — 방향성 참고용.",
        "- ⚠️ 이 실험은 *주어진 레이블을 잘 랭킹하나*만 측정. **레이블 정의 자체의 타당성은 별개 문제**.",
    ]
    (OUT / "ap_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n저장: {OUT / 'ap_summary.md'}")


# ── main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("AP 실험 — 불균형 대응 기법 비교")
    print("=" * 60)
    if not SRC.exists():
        raise FileNotFoundError(f"{SRC} 없음 — run_transition_pipeline.py 먼저 실행")
    if not HAS_IMBLEARN:
        print("  [경고] imbalanced-learn 미설치 → GB_SMOTE 조건 건너뜀")

    df = pd.read_csv(SRC, encoding="utf-8-sig")
    feat_cols = [c for c in df.columns if any(c.startswith(p) for p in FEAT_PREFIXES)]
    print(f"  피처 {len(feat_cols)}개: {feat_cols}")

    all_rows, chances = [], {}
    for label in LABELS:
        sub = df.dropna(subset=[label]).copy()
        X = sub[feat_cols].to_numpy(dtype=float)
        y = sub[label].astype(int).to_numpy()
        chance = int(y.sum()) / len(y)
        chances[label] = chance
        print(f"\n{'='*60}\n레이블: {label}  |  N={len(y)}  양성={int(y.sum())} ({chance:.1%})\n{'='*60}")
        for n_rep in N_REPEATS_LIST:
            print(f"\n  ── n_repeats={n_rep} ──")
            summary = run_label(X, y, label, n_rep, chance, save_plot=(n_rep == PRIMARY_REPEAT))
            all_rows.append(summary)

    all_summary = pd.concat(all_rows, ignore_index=True)
    write_report(all_summary, chances)

    print("\n" + "=" * 60)
    print("완료! 결과: Outputs/ap_experiment/ap_summary.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
