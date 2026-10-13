"""Paper figures 1 to 3 from outputs/ (receptor values per donor from the $WORK profiles).
Figure 1: leave-one-donor-out cycle clocks per compartment (snRNA-seq), treated donors alongside.
Figure 2: stroma clock against epithelium clock with the natural-donor line, and position-matched epithelial residuals,
progestin-only (snRNA-seq) and combined pill (scRNA-seq).
Figure 3: response of cycle genes per compartment, epithelial window-of-implantation score, epithelial PGR and ESR1.
Figure titles are in the legends, not in the images. One layout pixel is 0.5 pt at print size (PNG at 600 dpi), so the
16 px fonts print at 8 pt and the widest figures (960 px) at 169 mm.
Writes figures/figure<k>.{html,pdf,png}. Usage: figures.py"""
import json, os
import numpy as np, pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from s2_clock import treatment_group, W, MIN_N
from s5_mechanism import label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, FIG = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "figures")
C = pio.templates["plotly_white"].layout.colorway
GROUPS = ["proliferative", "secretory_early", "secretory_mid", "treated"]
NAME = {"proliferative": "Proliferative", "secretory_early": "Early secretory", "secretory_mid": "Mid-secretory", "other": "Natural, other stage"}
FS = 16  # 8 pt at print size
COLOR = {"proliferative": C[0], "secretory_early": C[2], "secretory_mid": C[4], "treated": C[1], "other": C[7]}
SETS = {"nuclei": ("progestin_only", "Progestin only", "snRNA-seq"), "cells_combined_pill": ("combined_pill", "Combined pill", "scRNA-seq")}


def ax(k, a):
    return f"{a}{'' if k == 1 else k} domain"


def ptxt(p):
    """P value as in the text, powers of ten below 0.001."""
    if p < 0.001:
        m, e = f"{p:.1e}".split("e")
        return f"P = {m} × 10" + str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"P = {p:.2g}"


def headroom(fig, r, c, v):
    """Space above the data for the P value in the top right corner."""
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v)); d = hi - lo
    fig.update_yaxes(range=[lo - 0.08 * d, hi + 0.28 * d], row=r, col=c)


def jitter(n, w=0.12, seed=0):
    return np.random.default_rng(seed).uniform(-w, w, n)


def save(fig, k, w, h, t=40):
    os.makedirs(FIG, exist_ok=True)
    fig.update_layout(template="plotly_white", width=w, height=h, font=dict(size=FS), margin=dict(l=70, r=15, t=t, b=40))
    fig.update_annotations(font_size=FS)
    fig.update_xaxes(title_font_size=FS, tickfont_size=FS, tickangle=0); fig.update_yaxes(title_font_size=FS, tickfont_size=FS)
    fig.write_html(os.path.join(FIG, f"figure{k}.html"), include_plotlyjs="cdn")
    fig.write_image(os.path.join(FIG, f"figure{k}.pdf")); fig.write_image(os.path.join(FIG, f"figure{k}.png"), scale=600 / 144)


def lab_of(d, treated_name):
    st = d.stage.astype(str) if "stage" in d else d.Stage.astype(str)
    g = np.where(d.group != "natural", np.where(d.group == treated_name, "treated", "skip"),
                 np.where(st.str.startswith("Proliferative"), "proliferative", np.where(st == "Secretory Early", "secretory_early",
                          np.where(st == "Secretory Mid", "secretory_mid", "other"))))
    return pd.Series(g, index=d.index)


def strip(fig, r, c, y, groups, labels, showlegend, tname):
    for i, g in enumerate(labels):
        v = y[groups == g].dropna()
        fig.add_trace(go.Box(x=np.full(len(v), i) + 0.0, y=v, name=tname if g == "treated" else NAME[g], marker_color=COLOR[g],
                             boxpoints=False, line=dict(color=COLOR[g]), fillcolor="rgba(0,0,0,0)", showlegend=False, width=0.5), r, c)
        fig.add_trace(go.Scatter(x=i + jitter(len(v), seed=i), y=v, mode="markers", marker=dict(color=COLOR[g], size=8, opacity=0.8, line=dict(width=0)),
                                 name=tname if g == "treated" else NAME[g], legendgroup=g, showlegend=showlegend), r, c)
    fig.update_xaxes(tickvals=list(range(len(labels))), ticktext=[(tname if g == "treated" else NAME[g]).replace(" ", "<br>") for g in labels], row=r, col=c)
    return y[groups.isin(labels)]


def figure1():
    d = pd.read_csv(os.path.join(OUT, "s2_nuclei_donors.csv")); s = json.load(open(os.path.join(OUT, "s2_nuclei_summary.json")))
    comps = ["stroma", "epithelium", "endothelium", "immune"]
    fig = make_subplots(2, 2, shared_yaxes=True, horizontal_spacing=0.05, vertical_spacing=0.16,
                        subplot_titles=[f"{c.capitalize()}, rho {s['pc1'][c]['rho']:.2f}, {s['pc1'][c]['donors_train']} donors" for c in comps])
    for j, comp in enumerate(comps):
        x = d[d.compartment == comp]; g = lab_of(x, "progestin_only")
        strip(fig, j // 2 + 1, j % 2 + 1, x.clock, g, GROUPS, False, "Progestin only")
    for v in (0, 1, 2):
        fig.add_hline(y=v, line=dict(color="lightgrey", dash="dot", width=1))
    for r in (1, 2):
        fig.update_yaxes(title_text="Cycle stage (clock)", row=r, col=1)
    save(fig, 1, 960, 720)


def figure2():
    fig = make_subplots(2, 2, row_heights=[0.6, 0.4], vertical_spacing=0.17, horizontal_spacing=0.12,
                        subplot_titles=[f"{v[1]} ({v[2]})" for v in SETS.values()] + ["", ""])
    for j, (tag, (tg, tname, tech)) in enumerate(SETS.items(), 1):
        d = pd.read_csv(os.path.join(OUT, f"s4_{tag}_donors.csv")); s = json.load(open(os.path.join(OUT, f"s4_{tag}_summary.json")))["main"]
        g = lab_of(d, tg); nat = d[d.group == "natural"]
        b = np.polyfit(nat.stroma, nat.epithelium, 1); xs = np.linspace(d.stroma.min(), d.stroma.max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=np.polyval(b, xs), mode="lines", line=dict(color="lightgrey", width=2), name="Natural line",
                                 showlegend=j == 1), 1, j)
        for k in ["other"] + GROUPS:
            x = d[g == k]
            fig.add_trace(go.Scatter(x=x.stroma, y=x.epithelium, mode="markers", name=NAME.get(k, "Hormone-exposed"), legendgroup=k, showlegend=j == 1,
                                     marker=dict(color=COLOR[k], size=11 if k == "treated" else 8, symbol="diamond" if k == "treated" else "circle",
                                                 line=dict(width=0), opacity=0.85),
                                     text=x.get("Hormonal treatment"), hovertemplate="%{text}<br>stroma %{x:.2f}<br>epithelium %{y:.2f}"), 1, j)
        fig.update_xaxes(title_text="Stromal clock", row=1, col=j); fig.update_yaxes(title_text="Epithelial clock" if j == 1 else "", row=1, col=j)
        gg = pd.Series(np.where(g == "treated", "treated", np.where(d.group == "natural", "natural", "skip")), index=d.index)
        for i, k in enumerate(["natural", "treated"]):
            v = d.residual[gg == k].dropna(); col = C[7] if k == "natural" else C[1]
            fig.add_trace(go.Box(x=np.full(len(v), i), y=v, boxpoints=False, line=dict(color=col), fillcolor="rgba(0,0,0,0)", showlegend=False, width=0.5), 2, j)
            fig.add_trace(go.Scatter(x=i + jitter(len(v), seed=i), y=v, mode="markers", showlegend=False,
                                     marker=dict(color=col, size=8, opacity=0.8, line=dict(width=0))), 2, j)
        fig.add_hline(y=0, line=dict(color="lightgrey", dash="dot", width=1), row=2, col=j)
        fig.update_xaxes(tickvals=[0, 1], ticktext=[f"Natural (n = {s['n_natural']})", f"{tname} (n = {s['n_treated']})"], row=2, col=j)
        fig.update_yaxes(title_text="Distance from<br>the natural line" if j == 1 else "", row=2, col=j)
        fig.add_annotation(text=f"{ptxt(s['p_s1_vs_natural'])}, permutation {ptxt(s['perm_s1'])}", showarrow=False,
                           xref=ax(2 + j, "x"), yref=ax(2 + j, "y"), x=0.5, y=1.02, xanchor="center", yanchor="bottom")
    fig.update_layout(legend=dict(orientation="h", y=-0.1))
    save(fig, 2, 920, 820)


def donor_values(s, tg, genes):
    P = pd.read_parquet(os.path.join(W, f"profiles_{s}.parquet"), columns=genes)
    M = pd.read_csv(os.path.join(W, f"profiles_{s}_meta.csv"), index_col=0).loc[P.index]
    M["group"] = M["Hormonal treatment"].map(treatment_group); M["lab"] = label(M, tg)
    m = (M.compartment == "epithelium") & (M.n >= MIN_N)
    return P[m].assign(lab=M.lab[m])


def figure3():
    titles = [f"{v[1]} ({v[2]})" for v in SETS.values()] + [""] * 6
    fig = make_subplots(4, 2, vertical_spacing=0.07, horizontal_spacing=0.12, subplot_titles=titles)
    for r, (tag, (tg, tname, tech)) in enumerate(SETS.items(), 1):  # r is the column here
        G = pd.read_csv(os.path.join(OUT, f"s5_{tag}_genes.csv")); G = G[G.direction == "up"]
        sm = json.load(open(os.path.join(OUT, f"s5_{tag}_summary.json")))
        for i, comp in enumerate(["stroma", "epithelium"]):
            v = G.fraction[G.compartment == comp].clip(-1, 2); col = C[0] if comp == "stroma" else C[3]
            fig.add_trace(go.Violin(x=np.full(len(v), i), y=v, line=dict(color=col), fillcolor=col, opacity=0.5, points=False,
                                    box_visible=True, meanline_visible=False, showlegend=False, width=0.7), 1, r)
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Stroma", "Epithelium"], row=1, col=r)
        fig.add_annotation(text=ptxt(sm['axis']['p_up_epithelium_lags_stroma']), showarrow=False, xref=ax(r, "x"),
                           yref=ax(r, "y"), x=0.98, y=1.0, xanchor="right", yanchor="top")
        s2 = "s2_nuclei_donors.csv" if tag == "nuclei" else "s2_cells_combined_pill_donors.csv"
        d = pd.read_csv(os.path.join(OUT, s2)); d = d[d.compartment == "epithelium"]
        strip(fig, 2, r, d.woi_score, lab_of(d, tg), ["proliferative", "secretory_mid", "treated"], False, tname)
        E = donor_values("nuclei" if tag == "nuclei" else "cells", tg, ["PGR", "ESR1"])
        for c, gene in ((3, "PGR"), (4, "ESR1")):
            headroom(fig, c, r, strip(fig, c, r, E[gene], E.lab, ["proliferative", "secretory_mid", "treated"], False, tname))
            rec = sm["receptors"]["epithelium"][gene]
            fig.add_annotation(text=ptxt(rec['p_vs_mid']), showarrow=False, xref=ax(2 * (c - 1) + r, "x"),
                               yref=ax(2 * (c - 1) + r, "y"), x=0.98, y=1.0, xanchor="right", yanchor="top")
    for c, t in enumerate(("Share of change<br>reached", "Epithelial<br>WOI score", "Epithelial <i>PGR</i><br>(log expression)",
                           "Epithelial <i>ESR1</i><br>(log expression)"), 1):
        fig.update_yaxes(title_text=t, row=c, col=1)
    save(fig, 3, 900, 1160, t=50)


GCOL = {"Natural": C[0], "Control": C[0], "Cu-IUD": C[5], "Programmed": C[1], "LNG-IUS": C[1], "DMPA": C[3], "COC": C[9],
        "LH+2": C[0], "LH+8": C[4]}


def strip2(fig, r, c, v, g, order, p=None, ncol=3):
    for i, k in enumerate(order):
        y = v[g == k].dropna()
        fig.add_trace(go.Box(x=np.full(len(y), i), y=y, boxpoints=False, line=dict(color=GCOL[k]), fillcolor="rgba(0,0,0,0)",
                             showlegend=False, width=0.5), r, c)
        fig.add_trace(go.Scatter(x=i + jitter(len(y), seed=i), y=y, mode="markers", showlegend=False,
                                 marker=dict(color=GCOL[k], size=8, opacity=0.8, line=dict(width=0))), r, c)
    fig.update_xaxes(tickvals=list(range(len(order))), ticktext=[f"{k}<br>({int(v[g == k].notna().sum())})" for k in order], row=r, col=c)
    headroom(fig, r, c, v[g.isin(order)])
    if p is not None:
        k = (r - 1) * ncol + c
        fig.add_annotation(text=ptxt(p), showarrow=False, xref=ax(k, "x"), yref=ax(k, "y"), x=0.98, y=1.0, xanchor="right", yanchor="top")


def figure4():
    h = json.load(open(os.path.join(OUT, "s6_huang_summary.json"))); b = json.load(open(os.path.join(OUT, "s7_bulk_summary.json")))
    H = pd.read_csv(os.path.join(OUT, "s6_huang_donors.csv")); H["g"] = H.cycle.replace({"Controlled": "Programmed"})
    S = pd.read_csv(os.path.join(OUT, "s7_bulk_samples.csv"))
    titles = ["Index, pre-specified scale", "Index, within-set scale", "Epithelial WOI score",
              "Stroma minus epithelium index", "Epithelial clock", "Epithelial WOI score",
              "Stroma minus epithelium index", "Epithelial clock", "Epithelial WOI score"]
    fig = make_subplots(3, 3, vertical_spacing=0.12, horizontal_spacing=0.05, subplot_titles=titles)
    u, z = H[H.version == "unscaled"], H[H.version == "zscored"]; o = ["Natural", "Programmed"]
    strip2(fig, 1, 1, u.dyssynchrony, u.g, o, h["unscaled"]["p_e1_dyssynchrony_higher"])
    strip2(fig, 1, 2, z.dyssynchrony, z.g, o, h["zscored"]["p_e1_dyssynchrony_higher"])
    strip2(fig, 1, 3, u.woi, u.g, o, h["unscaled"]["p_e2_woi_lower"])
    for r, (gse, o, lab) in enumerate((("GSE60129", ["control", "LNG-IUS", "DMPA"], "GSE60129"),
                                      ("GSE137765", ["control", "Cu-IUD", "LNG-IUS", "COC"], "GSE137765")), 2):
        d = S[S.set == gse].assign(group=lambda x: x.group.replace({"control": "Control"})); sm = b[gse]; o = [k if k != "control" else "Control" for k in o]
        strip2(fig, r, 1, d.dyssynchrony, d.group, o, sm["p_dyssynchrony_higher"])
        strip2(fig, r, 2, d.epithelium, d.group, o, sm["p_epithelium_two_sided"])
        strip2(fig, r, 3, d.woi, d.group, o, sm["p_woi_lower"])
    for r, t in ((1, "Huang 2026<br>(snRNA-seq)"), (2, "GSE60129<br>(microarray)"), (3, "GSE137765<br>(microarray)")):
        fig.update_yaxes(title_text=t, row=r, col=1)
    save(fig, 4, 960, 1000, t=50)


def figure_s1():
    S = pd.read_csv(os.path.join(OUT, "s7_bulk_samples.csv")); S = S[S.set == "GSE97929"]
    b = json.load(open(os.path.join(OUT, "s7_bulk_summary.json")))["GSE97929_transfer"]
    S["g"] = np.where(S.group == "LH2", "LH+2", "LH+8")
    fig = make_subplots(1, 2, subplot_titles=["Sorted stroma", "Sorted epithelium"], horizontal_spacing=0.12)
    for c, comp in enumerate(("stroma", "epithelium"), 1):
        d = S[S[comp].notna()]
        strip2(fig, 1, c, d[comp], d.g, ["LH+2", "LH+8"], b[comp]["p_lh8_higher"], ncol=2)
    fig.update_yaxes(title_text="Clock, within-set z-scores", row=1, col=1)
    save(fig, "S3", 640, 400)


def figure_s2():
    rows = []
    for tag, nm in (("nuclei", "Progestin only (snRNA-seq)"), ("cells_combined_pill", "Combined pill (scRNA-seq)")):
        s = json.load(open(os.path.join(OUT, f"s4_{tag}_summary.json")))
        items = [("Main", s["main"]), ("No endometriosis", s["s2_no_endometriosis"]), ("Endometriosis only", s["s2_endometriosis_only"])]
        items += [(f"Without {k}" if len(k) < 20 else "Without norethindrone acetate pill", v) for k, v in s.get("s3_leave_one_drug_out", {}).items()]
        items += [({"hvg1000": "1,000 genes", "hvg5000": "5,000 genes", "min100": "At least 100 nuclei", "specific_genes": "Compartment-specific genes"}[k], v)
                  for k, v in s["s5"].items()]
        for a, v in items:
            if "p_s1_vs_natural" in v:
                rows.append(dict(set=nm, analysis=a, n=v["n_treated"], residual=v["residual_median_treated"], p=v["p_s1_vs_natural"]))
    R = pd.DataFrame(rows); R["label"] = R.analysis + " (n = " + R.n.astype(str) + ")"
    fig = make_subplots(2, 1, subplot_titles=list(R.set.unique()), vertical_spacing=0.1)
    for c, (nm, d) in enumerate(R.groupby("set", sort=False), 1):  # c is the row here
        col = [C[1] if p < 0.05 else C[7] for p in d.p]
        fig.add_trace(go.Scatter(x=d.residual, y=d.label, mode="markers+text", text=[ptxt(p) for p in d.p], textposition="middle right",
                                 marker=dict(color=col, size=10, line=dict(width=0)), showlegend=False), c, 1)
        fig.add_vline(x=0, line=dict(color="lightgrey", dash="dot"), row=c, col=1)
        fig.update_xaxes(range=[-0.8, 0.5], row=c, col=1)
        fig.update_yaxes(autorange="reversed", row=c, col=1)
    fig.update_xaxes(title_text="Median distance from the natural line", row=2, col=1)
    save(fig, "S2", 900, 900, t=50)


def figure_s3():
    R = json.load(open(os.path.join(OUT, "s9_review.json")))
    fig = make_subplots(1, 2, subplot_titles=["Progestin only (snRNA-seq)", "Combined pill (scRNA-seq)"], horizontal_spacing=0.2)
    for c, tag in enumerate(("nuclei", "cells_combined_pill"), 1):
        a = R[tag]["r1_all_compartments"]; comps = [k for k in ("stroma", "epithelium", "immune", "endothelium") if "residual_treated" in a[k]]
        for g, key, col in (("Natural", "residual_natural", C[0]), ("Hormone-exposed", "residual_treated", C[1])):
            fig.add_trace(go.Scatter(x=[a[k][key] for k in comps], y=[f"{k.capitalize()} (n = {a[k]['n_treated']})" for k in comps], mode="markers",
                                     marker=dict(size=12, color=col, line=dict(width=0)), name=g, showlegend=c == 1), 1, c)
        fig.add_trace(go.Scatter(x=[0.95] * len(comps), y=[f"{k.capitalize()} (n = {a[k]['n_treated']})" for k in comps], mode="text",
                                 text=[ptxt(a[k]['p_two_sided']) for k in comps], textposition="middle left", showlegend=False), 1, c)
        fig.add_vline(x=0, line=dict(color="lightgrey", dash="dot"), row=1, col=c)
        fig.update_xaxes(title_text="Median difference from<br>the other compartments", range=[-0.6, 0.95], row=1, col=c)
        fig.update_yaxes(autorange="reversed", row=1, col=c)
    fig.update_layout(legend=dict(orientation="h", y=-0.35))
    save(fig, "S1", 960, 440)


def figure_s4():
    D = pd.read_csv(os.path.join(OUT, "s10_rank_bulk_samples.csv")); s = json.load(open(os.path.join(OUT, "s10_rank_bulk.json")))
    b = s["natural_line"]; xs = np.linspace(-0.6, 3.0, 20)
    fig = make_subplots(1, 3, subplot_titles=["Natural dated arrays", "GSE60129", "GSE137765"], horizontal_spacing=0.04, shared_yaxes=True)
    nat = D[D.group == "natural"]
    for c in (1, 2, 3):
        fig.add_trace(go.Scatter(x=xs, y=b["slope"] * xs + b["intercept"], mode="lines", line=dict(color="lightgrey", width=2), name="Natural line", showlegend=c == 1), 1, c)
    for ph, nm, col in ((0, "Proliferative", C[0]), (1, "Early secretory", C[2]), (2, "Mid-secretory", C[4])):
        x = nat[nat.phase == ph]
        fig.add_trace(go.Scatter(x=x.stroma, y=x.epithelium, mode="markers", name=nm, marker=dict(color=col, size=8, line=dict(width=0))), 1, 1)
    for c, (key, groups) in enumerate((("gse60129", [("control", "Control", C[6]), ("LNG-IUS", "LNG-IUS", C[1]), ("DMPA", "DMPA", C[3])]),
                                       ("gse137765", [("control", "Control", C[6]), ("cu-IUD", "Cu-IUD", C[5]), ("LNG-IUD", "LNG-IUS", C[1]), ("COC", "COC", C[9])])), 2):
        d = D[D.set == key]
        for g, nm, col in groups:
            x = d[d.group == g]
            fig.add_trace(go.Scatter(x=x.stroma, y=x.epithelium, mode="markers", name=nm, legendgroup=nm, showlegend=not (nm in ("Control", "LNG-IUS") and c == 3),
                                     marker=dict(color=col, size=8, line=dict(width=0))), 1, c)
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="Stromal clock", range=[-0.6, 3.0], row=1, col=c)
    fig.update_yaxes(title_text="Epithelial clock", row=1, col=1)
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    save(fig, "S4", 960, 460)


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure4(); figure_s1(); figure_s2(); figure_s3(); figure_s4()
    print(sorted(os.listdir(FIG)))
