import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

np.random.seed(42)

st.set_page_config(
    page_title="AML - Adversarial Attack & Defense Dashboard",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d0d1a 0%, #0a1628 50%, #0d1117 100%); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important; border-right: 1px solid #30363d; }
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="metric-container"] { background: linear-gradient(135deg, rgba(22,27,34,0.9) 0%, rgba(13,17,23,0.9) 100%); border: 1px solid #30363d; border-radius: 12px; padding: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
[data-testid="metric-container"] label { color: #8b949e !important; font-size:0.78rem; }
[data-testid="metric-container"] [data-testid="metric-value"] { color: #58a6ff !important; font-size:1.8rem; font-weight:700; }
h1 { color: #e6edf3 !important; font-weight: 700 !important; }
h2 { color: #58a6ff !important; font-weight: 600 !important; }
h3 { color: #79c0ff !important; font-weight: 500 !important; }
p, li { color: #8b949e !important; }
.stTabs [data-baseweb="tab-list"] { background: rgba(13,17,23,0.8); border-bottom: 2px solid #30363d; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #8b949e !important; border-radius: 8px 8px 0 0; padding: 8px 20px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: rgba(88,166,255,0.15) !important; color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }
hr { border-color: #30363d !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(13,17,23,0)",
    plot_bgcolor="rgba(22,27,34,0.6)",
    font=dict(family="Inter", color="#8b949e"),
    title_font=dict(color="#e6edf3", size=15),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d"),
    legend=dict(bgcolor="rgba(13,17,23,0.8)", bordercolor="#30363d", borderwidth=1),
    margin=dict(l=40, r=20, t=50, b=40),
)
ATTACK_COLORS = {
    "WB1_FGSM": "#ff6b6b",
    "WB2_PGD": "#ff9f43",
    "BB1_RandomSearch": "#48dbfb",
    "BB2_BoundaryAttack": "#ff6b9d",
    "BB3_SurrogateTransfer": "#a29bfe",
}
MODEL_COLORS = {
    "LogisticRegression": "#58a6ff",
    "SVM_RBF": "#3fb950",
    "RandomForest": "#d2a8ff",
    "MLP": "#ffa657",
}

@st.cache_data(show_spinner=False)
def load_and_prepare():
    data = load_wine()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, data.feature_names, data.target_names, data

@st.cache_resource(show_spinner=False)
def train_models(X_train, y_train):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42),
    }
    for m in models.values():
        m.fit(X_train, y_train)
    return models

def numerical_gradient(model, x, y_true, eps=1e-3):
    base = model.predict_proba(x.reshape(1, -1))[0][y_true]
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_p = x.copy()
        x_p[i] += eps
        p_plus = model.predict_proba(x_p.reshape(1, -1))[0][y_true]
        grad[i] = (p_plus - base) / eps
    return grad

def fgsm_attack(model, x, y_true, epsilon=0.5):
    return x - epsilon * np.sign(numerical_gradient(model, x, y_true))

def pgd_attack(model, x, y_true, epsilon=0.5, alpha=0.1, steps=10):
    x_adv = x.copy()
    for _ in range(steps):
        grad = numerical_gradient(model, x_adv, y_true)
        x_adv -= alpha * np.sign(grad)
        x_adv = np.clip(x_adv, x - epsilon, x + epsilon)
    return x_adv

def random_search_attack(model, x, y_true, epsilon=0.5, queries=200):
    rng = np.random.default_rng(1)
    best_x = x.copy()
    best_p = model.predict_proba(x.reshape(1, -1))[0][y_true]
    for _ in range(queries):
        cand = x + rng.uniform(-epsilon, epsilon, size=x.shape)
        p = model.predict_proba(cand.reshape(1, -1))[0][y_true]
        if p < best_p:
            best_p, best_x = p, cand
    return best_x

def boundary_attack(model, x, y_true, epsilon=1.5, steps=30):
    rng = np.random.default_rng(7)
    x_adv = x + rng.uniform(-epsilon, epsilon, size=x.shape)
    tries = 0
    while model.predict(x_adv.reshape(1, -1))[0] == y_true and tries < 20:
        x_adv = x + rng.uniform(-epsilon * 1.5, epsilon * 1.5, size=x.shape)
        tries += 1
    for _ in range(steps):
        mid = (x + x_adv) / 2
        if model.predict(mid.reshape(1, -1))[0] == y_true:
            x_adv = mid
    return x_adv

def surrogate_transfer_attack(target_model, x, y_true, X_train, y_train, epsilon=0.6):
    surrogate = LogisticRegression(max_iter=2000, random_state=42).fit(X_train, y_train)
    grad = numerical_gradient(surrogate, x, y_true)
    return x - epsilon * np.sign(grad)

ATTACK_FNS = {
    "WB1_FGSM": lambda m, x, yt, Xtr, ytr: fgsm_attack(m, x, yt),
    "WB2_PGD": lambda m, x, yt, Xtr, ytr: pgd_attack(m, x, yt),
    "BB1_RandomSearch": lambda m, x, yt, Xtr, ytr: random_search_attack(m, x, yt),
    "BB2_BoundaryAttack": lambda m, x, yt, Xtr, ytr: boundary_attack(m, x, yt),
    "BB3_SurrogateTransfer": lambda m, x, yt, Xtr, ytr: surrogate_transfer_attack(m, x, yt, Xtr, ytr),
}

def adversarial_training_defense(model_class, model_kwargs, X_train, y_train, frac=0.3):
    rng = np.random.default_rng(2)
    n_aug = int(len(X_train) * frac)
    idx = rng.choice(len(X_train), n_aug, replace=False)
    base = model_class(**model_kwargs).fit(X_train, y_train)
    X_adv_batch = np.array([fgsm_attack(base, X_train[i], y_train[i]) for i in idx])
    X_aug = np.vstack([X_train, X_adv_batch])
    y_aug = np.concatenate([y_train, y_train[idx]])
    return model_class(**model_kwargs).fit(X_aug, y_aug)

def decision_randomization_predict(model, x, noise_std=0.15, n_votes=7):
    rng = np.random.default_rng(3)
    votes = [model.predict((x + rng.normal(0, noise_std, size=x.shape)).reshape(1, -1))[0]
             for _ in range(n_votes)]
    vals, counts = np.unique(votes, return_counts=True)
    return vals[np.argmax(counts)]

def feature_squeeze(x, bins=20, lo=-4, hi=4):
    edges = np.linspace(lo, hi, bins)
    return edges[np.digitize(x, edges) - 1]

def rejection_predict(model, x, margin_thresh=0.15):
    proba = model.predict_proba(x.reshape(1, -1))[0]
    top1, top2 = np.sort(proba)[::-1][:2]
    if top1 - top2 < margin_thresh:
        return -1
    return model.predict(x.reshape(1, -1))[0]

def robust_clip_predict(model, x, X_train, k=0.1):
    lo, hi = X_train.min(axis=0) - k, X_train.max(axis=0) + k
    return model.predict(np.clip(x, lo, hi).reshape(1, -1))[0]

@st.cache_data(show_spinner=False)
def run_attacks(_models, _X_test, _y_test, _X_train, _y_train, n_sample=30):
    np.random.seed(42)
    idxs = np.random.choice(len(_y_test), n_sample, replace=False)
    results = []
    for mname, model in _models.items():
        for aname, afn in ATTACK_FNS.items():
            successes = 0
            for i in idxs:
                x, yt = _X_test[i], _y_test[i]
                if model.predict(x.reshape(1, -1))[0] != yt:
                    continue
                x_adv = afn(model, x, yt, _X_train, _y_train)
                if model.predict(x_adv.reshape(1, -1))[0] != yt:
                    successes += 1
            results.append((mname, aname, successes, n_sample))
    df = pd.DataFrame(results, columns=["Model", "Attack", "Successes", "Total"])
    df["ASR"] = (df["Successes"] / df["Total"]).round(3)
    return df, idxs

@st.cache_data(show_spinner=False)
def run_defenses(_models, _X_test, _y_test, _X_train, _y_train, _idxs):
    MODEL_CTOR = {
        "LogisticRegression": (LogisticRegression, dict(max_iter=2000, random_state=42)),
        "SVM_RBF": (SVC, dict(kernel="rbf", probability=True, random_state=42)),
        "RandomForest": (RandomForestClassifier, dict(n_estimators=150, random_state=42)),
        "MLP": (MLPClassifier, dict(hidden_layer_sizes=(32,16), max_iter=2000, random_state=42)),
    }
    DEFENSES = ["D1_AdversarialTraining","D2_DecisionRandomization",
                "D3_FeatureSqueezing","D4_ConfidenceRejection","D5_RobustClipping"]
    def_results = []
    for mname, model in _models.items():
        ctor, kwargs = MODEL_CTOR[mname]
        hardened = adversarial_training_defense(ctor, kwargs, _X_train, _y_train)
        for dname in DEFENSES:
            before = after = tested = 0
            for i in _idxs:
                x, yt = _X_test[i], _y_test[i]
                if model.predict(x.reshape(1,-1))[0] != yt:
                    continue
                x_adv = fgsm_attack(model, x, yt)
                tested += 1
                if model.predict(x_adv.reshape(1,-1))[0] != yt:
                    before += 1
                if dname == "D1_AdversarialTraining":
                    pred = hardened.predict(x_adv.reshape(1,-1))[0]
                elif dname == "D2_DecisionRandomization":
                    pred = decision_randomization_predict(model, x_adv)
                elif dname == "D3_FeatureSqueezing":
                    pred = model.predict(feature_squeeze(x_adv).reshape(1,-1))[0]
                elif dname == "D4_ConfidenceRejection":
                    pred = rejection_predict(model, x_adv)
                elif dname == "D5_RobustClipping":
                    pred = robust_clip_predict(model, x_adv, _X_train)
                if pred != yt:
                    after += 1
            if tested:
                def_results.append((mname, dname, tested, round(before/tested,3), round(after/tested,3)))
    return pd.DataFrame(def_results, columns=["Model","Defense","Tested","ASR_before","ASR_after"])

@st.cache_data(show_spinner=False)
def get_epsilon_sweep(_models, _X_test, _y_test, _X_train, _y_train, _idxs):
    epsilons = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
    rows = []
    for eps in epsilons:
        for mname, model in _models.items():
            suc = tot = 0
            for i in _idxs[:20]:
                x, yt = _X_test[i], _y_test[i]
                if model.predict(x.reshape(1,-1))[0] != yt:
                    continue
                x_adv = fgsm_attack(model, x, yt, epsilon=eps)
                tot += 1
                if model.predict(x_adv.reshape(1,-1))[0] != yt:
                    suc += 1
            if tot:
                rows.append({"epsilon": eps, "Model": mname, "ASR": round(suc/tot,3)})
    return pd.DataFrame(rows)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px'>
      <div style='font-size:2.5rem'>shield</div>
      <div style='color:#58a6ff; font-weight:700; font-size:1.1rem'>AML Dashboard</div>
      <div style='color:#8b949e; font-size:.75rem; margin-top:4px'>CIA-3 - Decision-Time Evasion</div>
    </div>
    <hr style='border-color:#30363d; margin:12px 0'>
    """, unsafe_allow_html=True)

    st.markdown("**Controls**")
    n_sample = st.slider("Attack sample size", 10, 45, 30, 5)
    selected_models = st.multiselect(
        "Models to show",
        ["LogisticRegression","SVM_RBF","RandomForest","MLP"],
        default=["LogisticRegression","SVM_RBF","RandomForest","MLP"]
    )
    if not selected_models:
        selected_models = ["LogisticRegression","SVM_RBF","RandomForest","MLP"]

# ============================================================
# LOAD + TRAIN
# ============================================================
with st.spinner("Loading Wine dataset and training models..."):
    X_train, X_test, y_train, y_test, feat_names, target_names, wine_data = load_and_prepare()
    models = train_models(X_train, y_train)

with st.spinner("Running adversarial attacks (cached after first run)..."):
    attack_df, idxs = run_attacks(models, X_test, y_test, X_train, y_train, n_sample)

with st.spinner("Evaluating defenses..."):
    defense_df = run_defenses(models, X_test, y_test, X_train, y_train, idxs)

with st.spinner("Computing epsilon sweep..."):
    eps_df = get_epsilon_sweep(models, X_test, y_test, X_train, y_train, idxs)

attack_df_f = attack_df[attack_df["Model"].isin(selected_models)]
defense_df_f = defense_df[defense_df["Model"].isin(selected_models)]
eps_df_f = eps_df[eps_df["Model"].isin(selected_models)]

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(31,111,235,.15) 0%, rgba(56,139,253,.05) 100%);
  border: 1px solid #1f6feb55; border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;'>
  <h1 style='margin:0; font-size:1.9rem; color:#e6edf3 !important'>
    Adversarial ML Attack and Defense Dashboard
  </h1>
  <p style='margin:8px 0 0; color:#8b949e !important; font-size:.95rem'>
    CIA-3 - Decision-Time Evasion Attacks on the Wine Dataset - 4 Models - 5 Attacks - 5 Defenses
  </p>
</div>
""", unsafe_allow_html=True)

# KPI row
baseline_accs = {n: accuracy_score(y_test, m.predict(X_test)) for n,m in models.items()}
best_model = max(baseline_accs, key=baseline_accs.get)
worst_attack = attack_df.groupby("Attack")["ASR"].mean().idxmax()
avg_asr = attack_df_f["ASR"].mean()
best_defense = (defense_df.assign(reduction=lambda d: d.ASR_before - d.ASR_after)
                .groupby("Defense")["reduction"].mean().idxmax())

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Best Model", best_model.replace("Regression","Reg."), f"{baseline_accs[best_model]:.1%}")
k2.metric("Models Tested", "4", "LR + SVM + RF + MLP")
k3.metric("Avg. ASR", f"{avg_asr:.1%}", "across all attacks")
k4.metric("Deadliest Attack", worst_attack, "highest avg. ASR")
k5.metric("Best Defense", best_defense.replace("D1_","").replace("D2_","")
          .replace("D3_","").replace("D4_","").replace("D5_",""), "most ASR reduction")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tabs = st.tabs(["Baseline", "Attack Analysis", "Defense Analysis", "Epsilon Sweep", "Data Explorer"])

# ---- TAB 1: BASELINE ----
with tabs[0]:
    st.markdown("## Baseline Model Performance")
    col_a, col_b = st.columns([1,1])

    with col_a:
        accs = {n: accuracy_score(y_test, m.predict(X_test)) for n,m in models.items() if n in selected_models}
        fig = go.Figure()
        for mname, acc in accs.items():
            fig.add_trace(go.Bar(x=[mname], y=[acc], marker_color=MODEL_COLORS[mname],
                text=[f"{acc:.1%}"], textposition="outside", name=mname, width=0.5))
        fig.update_layout(**PLOTLY_LAYOUT, title="Clean Accuracy per Model",
                          yaxis_range=[0,1.15], showlegend=False, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        cm_model = st.selectbox("Model for confusion matrix", selected_models, key="cm_model")
        cm = confusion_matrix(y_test, models[cm_model].predict(X_test))
        fig_cm = px.imshow(cm, text_auto=True,
            color_continuous_scale=[[0,"#0d1117"],[0.5,"#1f6feb"],[1,"#58a6ff"]],
            labels=dict(x="Predicted", y="Actual"),
            x=[f"Class {i}" for i in range(3)], y=[f"Class {i}" for i in range(3)])
        fig_cm.update_layout(**PLOTLY_LAYOUT, title=f"Confusion Matrix - {cm_model}", height=340)
        st.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("Detailed Classification Reports"):
        for mname in selected_models:
            st.markdown(f"**{mname}**")
            report = classification_report(y_test, models[mname].predict(X_test),
                target_names=target_names, output_dict=True)
            df_rep = pd.DataFrame(report).T.round(3)
            st.dataframe(df_rep, use_container_width=True)
            st.markdown("---")

    st.markdown("### PCA Projection (2D)")
    pca_model_sel = st.selectbox("Model for PCA viz", selected_models, key="pca_model")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(np.vstack([X_train, X_test]))
    Xte2 = X_2d[len(X_train):]
    y_pred_te = models[pca_model_sel].predict(X_test)
    class_colors = ["#1f6feb","#3fb950","#d2a8ff"]
    fig_pca = go.Figure()
    for ci in range(3):
        mask = y_test == ci
        correct = mask & (y_pred_te == ci)
        wrong = mask & (y_pred_te != ci)
        if correct.any():
            fig_pca.add_trace(go.Scatter(x=Xte2[correct,0], y=Xte2[correct,1], mode="markers",
                marker=dict(size=9, color=class_colors[ci], symbol="circle",
                            line=dict(width=1, color="#e6edf3")), name=f"Class {ci} Correct"))
        if wrong.any():
            fig_pca.add_trace(go.Scatter(x=Xte2[wrong,0], y=Xte2[wrong,1], mode="markers",
                marker=dict(size=12, color=class_colors[ci], symbol="x",
                            line=dict(width=2, color="#ff6b6b")), name=f"Class {ci} Wrong"))
    fig_pca.update_layout(**PLOTLY_LAYOUT, title=f"PCA Projection - {pca_model_sel}",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", height=420)
    st.plotly_chart(fig_pca, use_container_width=True)

# ---- TAB 2: ATTACK ANALYSIS ----
with tabs[1]:
    st.markdown("## Attack Success Rate Analysis")
    col1, col2 = st.columns([1.3, 1])

    with col1:
        pivot = attack_df_f.pivot(index="Attack", columns="Model", values="ASR")
        fig_hm = px.imshow(pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            color_continuous_scale=[[0,"#0d1117"],[0.4,"#1f6feb"],[0.7,"#ff9f43"],[1,"#ff6b6b"]],
            text_auto=".2f", zmin=0, zmax=1, labels=dict(color="ASR"))
        fig_hm.update_layout(**PLOTLY_LAYOUT, title="ASR Heatmap (0=immune, 1=fully vulnerable)", height=380)
        st.plotly_chart(fig_hm, use_container_width=True)

    with col2:
        radar_model = st.selectbox("Model for radar chart", selected_models, key="radar_model")
        sub_df = attack_df_f[attack_df_f["Model"]==radar_model]
        cats = sub_df["Attack"].tolist()
        vals = sub_df["ASR"].tolist()
        cats_c = cats + [cats[0]]; vals_c = vals + [vals[0]]
        fig_rad = go.Figure(go.Scatterpolar(r=vals_c, theta=cats_c, fill="toself",
            line_color="#58a6ff", fillcolor="rgba(88,166,255,0.15)"))
        fig_rad.update_layout(**PLOTLY_LAYOUT,
            polar=dict(bgcolor="rgba(22,27,34,0.6)",
                radialaxis=dict(range=[0,1], tickcolor="#8b949e", gridcolor="#21262d"),
                angularaxis=dict(tickcolor="#8b949e", gridcolor="#21262d")),
            title=f"Attack Profile - {radar_model}", height=380)
        st.plotly_chart(fig_rad, use_container_width=True)

    st.markdown("### All Attacks vs All Models (Grouped Bar)")
    fig_gb = go.Figure()
    for aname in attack_df_f["Attack"].unique():
        sub = attack_df_f[attack_df_f["Attack"]==aname]
        fig_gb.add_trace(go.Bar(name=aname, x=sub["Model"], y=sub["ASR"],
            marker_color=ATTACK_COLORS.get(aname,"#8b949e"),
            text=sub["ASR"].apply(lambda v:f"{v:.1%}"), textposition="outside"))
    fig_gb.update_layout(**PLOTLY_LAYOUT, barmode="group", yaxis_range=[0,1.2],
        height=400, title="Attack Success Rate by Model and Attack Type")
    st.plotly_chart(fig_gb, use_container_width=True)

    st.markdown("### ASR Sunburst - Hierarchy View")
    fig_sun = px.sunburst(attack_df_f, path=["Model","Attack"], values="ASR", color="ASR",
        color_continuous_scale=[[0,"#1f6feb"],[0.5,"#ff9f43"],[1,"#ff6b6b"]])
    fig_sun.update_layout(**PLOTLY_LAYOUT, height=500,
        title="ASR Sunburst (inner=Model, outer=Attack)")
    st.plotly_chart(fig_sun, use_container_width=True)

    with st.expander("Raw Attack Results Table"):
        st.dataframe(attack_df_f, use_container_width=True)

# ---- TAB 3: DEFENSE ANALYSIS ----
with tabs[2]:
    st.markdown("## Defense Effectiveness Analysis (vs FGSM)")
    defense_df_f = defense_df_f.copy()
    defense_df_f["ASR_reduction"] = (defense_df_f["ASR_before"] - defense_df_f["ASR_after"]).round(3)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        def_model = st.selectbox("Select model", selected_models, key="def_model")
        sub = defense_df_f[defense_df_f["Model"]==def_model]
        fig_def = go.Figure()
        fig_def.add_trace(go.Bar(name="Before Defense", x=sub["Defense"], y=sub["ASR_before"],
            marker_color="#ff6b6b", text=sub["ASR_before"].apply(lambda v:f"{v:.1%}"), textposition="outside"))
        fig_def.add_trace(go.Bar(name="After Defense", x=sub["Defense"], y=sub["ASR_after"],
            marker_color="#3fb950", text=sub["ASR_after"].apply(lambda v:f"{v:.1%}"), textposition="outside"))
        fig_def.update_layout(**PLOTLY_LAYOUT, barmode="group", yaxis_range=[0,1.2],
            title=f"ASR Before vs After Defense - {def_model}", height=380)
        st.plotly_chart(fig_def, use_container_width=True)

    with col2:
        pivot_red = defense_df_f.pivot(index="Defense", columns="Model", values="ASR_reduction")
        fig_red = px.imshow(pivot_red.values, x=pivot_red.columns.tolist(), y=pivot_red.index.tolist(),
            color_continuous_scale=[[0,"#ff6b6b"],[0.5,"#0d1117"],[1,"#3fb950"]],
            text_auto=".2f", zmin=-0.5, zmax=0.5, labels=dict(color="ASR reduction"))
        fig_red.update_layout(**PLOTLY_LAYOUT, title="ASR Reduction Heatmap (green=effective)", height=380)
        st.plotly_chart(fig_red, use_container_width=True)

    st.markdown("### Defense Reduction Lollipop Chart")
    fig_lol = go.Figure()
    y_labels = []
    for mname in selected_models:
        sub = defense_df_f[defense_df_f["Model"]==mname].sort_values("ASR_reduction")
        for _, row in sub.iterrows():
            label = f"{mname} | {row['Defense']}"
            y_labels.append(label)
            color = "#3fb950" if row["ASR_reduction"] >= 0 else "#ff6b6b"
            fig_lol.add_trace(go.Scatter(x=[row["ASR_reduction"]], y=[label], mode="markers",
                marker=dict(size=14, color=color), name=label, showlegend=False))
            fig_lol.add_shape(type="line", x0=0, x1=row["ASR_reduction"], y0=label, y1=label,
                line=dict(color=color, width=2))
    fig_lol.add_vline(x=0, line_dash="dash", line_color="#8b949e", line_width=1)
    fig_lol.update_layout(**PLOTLY_LAYOUT, xaxis_title="ASR Reduction (positive = better defense)",
        title="Defense Reduction Lollipop Chart", height=max(420, len(y_labels)*30))
    st.plotly_chart(fig_lol, use_container_width=True)

    with st.expander("Raw Defense Results Table"):
        st.dataframe(defense_df_f, use_container_width=True)

# ---- TAB 4: EPSILON SWEEP ----
with tabs[3]:
    st.markdown("## Perturbation Budget (epsilon) vs Attack Success Rate")
    st.markdown("This sweep shows how increasing FGSM perturbation budget trades off against ASR - a fundamental robustness curve.")

    fig_eps = px.line(eps_df_f, x="epsilon", y="ASR", color="Model", markers=True,
        color_discrete_map=MODEL_COLORS,
        labels={"epsilon":"Perturbation Budget epsilon", "ASR":"Attack Success Rate"})
    fig_eps.add_hline(y=0.5, line_dash="dot", line_color="#8b949e",
        annotation_text="50% ASR", annotation_position="right")
    fig_eps.update_traces(line_width=2.5, marker_size=8)
    fig_eps.update_layout(**PLOTLY_LAYOUT, title="FGSM: epsilon vs ASR Robustness Curve", height=430)
    st.plotly_chart(fig_eps, use_container_width=True)

    fig_area = px.area(eps_df_f, x="epsilon", y="ASR", color="Model", color_discrete_map=MODEL_COLORS)
    fig_area.update_layout(**PLOTLY_LAYOUT, title="FGSM: Area Under Robustness Curve", height=350)
    st.plotly_chart(fig_area, use_container_width=True)

    auc_rows = []
    for mname in selected_models:
        sub = eps_df_f[eps_df_f["Model"]==mname].sort_values("epsilon")
        if len(sub) >= 2:
            auc = float(np.trapz(sub["ASR"], sub["epsilon"]))
            auc_rows.append({"Model": mname, "AUC_vulnerability": round(auc, 3)})
    if auc_rows:
        auc_df = pd.DataFrame(auc_rows).sort_values("AUC_vulnerability")
        fig_auc = go.Figure(go.Bar(x=auc_df["Model"], y=auc_df["AUC_vulnerability"],
            marker_color=[MODEL_COLORS[m] for m in auc_df["Model"]],
            text=auc_df["AUC_vulnerability"], textposition="outside"))
        fig_auc.update_layout(**PLOTLY_LAYOUT, title="Integrated Vulnerability Score (lower = more robust)",
            height=320, showlegend=False)
        st.plotly_chart(fig_auc, use_container_width=True)

# ---- TAB 5: DATA EXPLORER ----
with tabs[4]:
    st.markdown("## Wine Dataset Explorer")
    wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    wine_df["Class"] = [wine_data.target_names[i] for i in wine_data.target]

    col1, col2 = st.columns(2)
    with col1:
        feat_x = st.selectbox("X axis feature", wine_data.feature_names, index=0)
    with col2:
        feat_y = st.selectbox("Y axis feature", wine_data.feature_names, index=6)

    fig_scatter = px.scatter(wine_df, x=feat_x, y=feat_y, color="Class",
        color_discrete_sequence=["#58a6ff","#3fb950","#d2a8ff"],
        marginal_x="histogram", marginal_y="box", symbol="Class",
        title=f"Wine Dataset: {feat_x} vs {feat_y}")
    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=480)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Random Forest Feature Importances")
    rf = models["RandomForest"]
    importances = pd.DataFrame({
        "Feature": wine_data.feature_names, "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig_imp = go.Figure(go.Bar(x=importances["Importance"], y=importances["Feature"],
        orientation="h",
        marker=dict(color=importances["Importance"], colorscale=[[0,"#1f6feb"],[1,"#d2a8ff"]]),
        text=importances["Importance"].round(3), textposition="outside"))
    fig_imp.update_layout(**PLOTLY_LAYOUT, title="Feature Importance (Random Forest)",
        xaxis_title="Importance", height=420, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### Feature Correlation Heatmap")
    corr = wine_df.drop("Class", axis=1).corr().round(2)
    fig_corr = px.imshow(corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        color_continuous_scale=[[0,"#ff6b6b"],[0.5,"#0d1117"],[1,"#58a6ff"]],
        text_auto=True, zmin=-1, zmax=1)
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=520, title="Pearson Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("Raw Dataset Preview"):
        st.dataframe(wine_df, use_container_width=True)

st.markdown("""
<hr style='border-color:#30363d; margin:32px 0 16px'>
<div style='text-align:center; color:#8b949e; font-size:.8rem'>
  CIA-3 - Adversarial Machine Learning - Decision-Time Evasion Attacks and Defenses<br>
  Built with Streamlit + Plotly + scikit-learn
</div>
""", unsafe_allow_html=True)
