"""Qrafik funksiyaları: həssaslıq, 3D səth və boru vizuallaşdırması (Plotly).

Bütün nöqtələr HybridModel.predict() ilə hesablanır, yəni qrafiklər tətbiqin
real hesablamasının özüdür. Boru vizuallaşdırması isə analitik illüstrasiyadır
(CFD deyil).
"""
import numpy as np
import plotly.graph_objects as go

VAR_INFO = {
    "velocity": ("Sürət v", "m/s"),
    "T_wall": ("Divar temperaturu T_wall", "°C"),
    "T_ref": ("Axın temperaturu T_ref", "°C"),
    "L": ("Xarakterik ölçü L (diametr və ya hündürlük)", "m"),
    "L_over_D": ("L/D", ""),
}
FORCED = {"pipe_CWT", "pipe_CHF", "plate_forced", "cylinder_cross", "sphere_forced"}
PIPES = {"pipe_CWT", "pipe_CHF"}


def available_vars(geometry):
    v = ["T_wall", "T_ref", "L"]
    if geometry in FORCED:
        v.insert(0, "velocity")
    if geometry in PIPES:
        v.append("L_over_D")
    return v


def var_label(var):
    name, unit = VAR_INFO[var]
    return f"{name}, {unit}" if unit else name


def to_internal(var, x):
    """Ekran qiyməti → daxili qiymət (°C → K)."""
    return np.asarray(x, dtype=float) + (273.15 if var in ("T_wall", "T_ref") else 0.0)


def to_display(var, x):
    return np.asarray(x, dtype=float) - (273.15 if var in ("T_wall", "T_ref") else 0.0)


def default_range(var, base, bounds):
    """Qrafik üçün ağlabatan standart diapazon (ekran vahidlərində). bounds=(Tmin, Tmax) K ilə."""
    tmin, tmax = bounds
    if var == "velocity":
        return 0.3 * base["velocity"], 3.0 * base["velocity"]
    if var == "T_wall":
        if base["T_wall"] >= base["T_ref"]:
            lo, hi = base["T_ref"] + 5, min(tmax - 1, base["T_wall"] + 30)
        else:
            lo, hi = max(tmin + 1, base["T_wall"] - 30), base["T_ref"] - 5
        return lo - 273.15, hi - 273.15
    if var == "T_ref":
        return max(tmin + 1, base["T_ref"] - 25) - 273.15, min(tmax - 1, base["T_ref"] + 25) - 273.15
    if var == "L":
        return 0.5 * base["L"], 2.0 * base["L"]
    return max(1.0, 0.25 * base["L_over_D"]), 2.0 * base["L_over_D"]


def _predict(model, base, **override):
    b = dict(base)
    b.update(override)
    try:
        return model.predict(b["geometry"], b["fluid"], T_ref=b["T_ref"], T_wall=b["T_wall"],
                             L=b["L"], velocity=b.get("velocity"),
                             L_over_D=b.get("L_over_D"), P=b["P"])
    except Exception:
        return None


def _point(out):
    """Bir nöqtənin göstəriciləri: Nu/h üçün fizika, yekun və interval."""
    if out is None:
        return {k: np.nan for k in ("Nu_phys", "Nu", "Nu_lo", "Nu_hi", "h_phys", "h", "h_lo", "h_hi")}
    r = out["result"]
    Nu, h = out["Nu"], out["h"]
    d = {"Nu_phys": out["Nu_physics"], "Nu": Nu, "h_phys": r.h, "h": h,
         "Nu_lo": np.nan, "Nu_hi": np.nan, "h_lo": np.nan, "h_hi": np.nan}
    if out["used_ml"] and out["interval"] is not None:
        lo, hi = out["interval"]
        d.update(Nu_lo=lo, Nu_hi=hi, h_lo=h * lo / Nu, h_hi=h * hi / Nu)
    return d


# ------------------------------------------------------------------ 1) 2D həssaslıq
def sweep_data(model, base, var, lo, hi, n=40):
    xs_disp = np.linspace(lo, hi, n)
    xs = to_internal(var, xs_disp)
    pts = [_point(_predict(model, base, **{var: float(x)})) for x in xs]
    data = {k: np.array([p[k] for p in pts]) for k in pts[0]}
    data["x"] = xs_disp
    data["failed"] = int(np.isnan(data["Nu"]).sum())
    return data


def sweep_figure(data, var, metric="Nu", logy=False, nu_name="Nu"):
    pre = "Nu" if metric == "Nu" else "h"
    ylab = nu_name if metric == "Nu" else "h, W/(m²·K)"
    x = data["x"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=data[pre + "_hi"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=data[pre + "_lo"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(99,110,250,0.2)",
                             name="90% interval (ML)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=data[pre + "_phys"], mode="lines",
                             line=dict(dash="dash"), name="Fizika düsturu"))
    fig.add_trace(go.Scatter(x=x, y=data[pre], mode="lines", line=dict(width=3),
                             name="Yekun (fizika + ML)"))
    fig.update_layout(xaxis_title=var_label(var), yaxis_title=ylab,
                      yaxis_type="log" if logy else "linear",
                      legend=dict(orientation="h", y=1.12), margin=dict(t=40))
    return fig


# ------------------------------------------------------------------ 2) 3D səth
def surface_data(model, base, xvar, yvar, xlim, ylim, n=15):
    xs_d = np.linspace(*xlim, n)
    ys_d = np.linspace(*ylim, n)
    Nu = np.full((n, n), np.nan)
    h = np.full((n, n), np.nan)
    for i, yv in enumerate(to_internal(yvar, ys_d)):
        for j, xv in enumerate(to_internal(xvar, xs_d)):
            p = _point(_predict(model, base, **{xvar: float(xv), yvar: float(yv)}))
            Nu[i, j], h[i, j] = p["Nu"], p["h"]
    return {"x": xs_d, "y": ys_d, "Nu": Nu, "h": h, "failed": int(np.isnan(Nu).sum())}


def surface_figure(data, xvar, yvar, metric="h", point=None, nu_name="Nu"):
    z = data[metric]
    zlab = "h, W/(m²·K)" if metric == "h" else nu_name
    fig = go.Figure(go.Surface(x=data["x"], y=data["y"], z=z, colorscale="Viridis",
                               colorbar=dict(title=zlab)))
    if point is not None:
        fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode="markers",
                                   marker=dict(size=6, color="red"), name="Cari giriş"))
    fig.update_layout(scene=dict(xaxis_title=var_label(xvar), yaxis_title=var_label(yvar),
                                 zaxis_title=zlab),
                      margin=dict(l=0, r=0, t=30, b=0), height=560, showlegend=False)
    return fig


# ------------------------------------------------------------------ 3) Boru vizuallaşdırması
def pipe_state(base, out):
    """Sabit divar temperaturu: T_b(x) = T_w - (T_w - T_giriş)·exp(-h·π·D·x/(ṁ·cp))."""
    v = out["result"].values
    D, U = base["L"], base["velocity"]
    Lp = base["L_over_D"] * D
    mdot = v["rho"] * U * np.pi * D ** 2 / 4
    Tin, Tw = base["T_ref"], base["T_wall"]
    x = np.linspace(0.0, Lp, 80)
    Tb = Tw - (Tw - Tin) * np.exp(-out["h"] * np.pi * D * x / (mdot * v["cp"]))
    laminar = v["Re"] < 2300
    umax = 2.0 * U if laminar else U / (2 * 49 / (8 * 15))   # 1/7 qüvvət qanunu: U/umax = 98/120
    return dict(D=D, Lp=Lp, x=x, Tb=Tb, Tin=Tin, Tw=Tw, Tout=Tb[-1],
                Q=mdot * v["cp"] * (Tb[-1] - Tin), mdot=mdot, laminar=laminar,
                umax=umax, Re=v["Re"])


def _u_norm(r_over_R, laminar):
    r = np.clip(r_over_R, 0, 1)
    return 1 - r ** 2 if laminar else (1 - r) ** (1 / 7)


def pipe_figure_3d(st_):
    R, Lp = st_["D"] / 2, st_["Lp"]
    th = np.linspace(0, 2 * np.pi, 48)
    X = np.outer(st_["x"], np.ones_like(th))
    Y = R * np.outer(np.ones_like(st_["x"]), np.cos(th))
    Z = R * np.outer(np.ones_like(st_["x"]), np.sin(th))
    C = np.outer(st_["Tb"] - 273.15, np.ones_like(th))
    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=C, colorscale="RdBu_r", opacity=0.6,
                             colorbar=dict(title="T_b, °C", x=1.0, len=0.5, y=0.75)))
    rr = np.linspace(0, R, 25)
    un = _u_norm(rr / R, st_["laminar"])
    Xp = np.outer(un, np.ones_like(th)) * 0.2 * Lp + Lp
    Yp = np.outer(rr, np.cos(th))
    Zp = np.outer(rr, np.sin(th))
    fig.add_trace(go.Surface(x=Xp, y=Yp, z=Zp, surfacecolor=np.outer(un, np.ones_like(th)),
                             colorscale="Viridis",
                             colorbar=dict(title="u/u_max", x=1.0, len=0.5, y=0.25)))
    fig.update_layout(scene=dict(xaxis_title="x, m", yaxis_title="y, m", zaxis_title="z, m",
                                 aspectmode="manual", aspectratio=dict(x=3, y=1, z=1)),
                      margin=dict(l=0, r=0, t=30, b=0), height=520)
    return fig


def pipe_figure_2d(st_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st_["x"], y=st_["Tb"] - 273.15, mode="lines",
                             line=dict(width=3), name="Orta temperatur T_b(x)"))
    fig.add_hline(y=st_["Tw"] - 273.15, line_dash="dash", annotation_text="Divar T_wall")
    fig.update_layout(xaxis_title="x, m", yaxis_title="T, °C", margin=dict(t=30), height=300)
    return fig
