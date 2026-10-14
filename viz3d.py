"""viz3d.py - 4 analitik həndəsə üçün 3D vizuallaşdırma (Plotly).
Fiziki xassələr (nu, alpha, k, Pr, beta) calculator-dakı CoolProp-dan verilir.
"""
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp, solve_bvp, trapezoid
from scipy.optimize import brentq

PIPE_TW = "Boru daxili axın, sabit divar temperaturu"
PIPE_QS = "Boru daxili axın, sabit istilik axını"
PLATE_FORCED = "Düz lövhə üzərində məcburi axın"
PLATE_NAT = "Şaquli lövhə, təbii konveksiya"
SUPPORTED = [PIPE_TW, PIPE_QS, PLATE_FORCED, PLATE_NAT]


def props(fluid, T_K, P_Pa):
    """CoolProp ilə xassələr."""
    from CoolProp.CoolProp import PropsSI as P
    rho = P("D", "T", T_K, "P", P_Pa, fluid)
    mu = P("V", "T", T_K, "P", P_Pa, fluid)
    k = P("L", "T", T_K, "P", P_Pa, fluid)
    cp = P("C", "T", T_K, "P", P_Pa, fluid)
    beta = P("ISOBARIC_EXPANSION_COEFFICIENT", "T", T_K, "P", P_Pa, fluid)
    return dict(nu=mu / rho, alpha=k / (rho * cp), k=k, Pr=mu * cp / k, beta=beta)


# ---------------- Analitik / oxşarlıq həlləri ----------------
def _graetz(n=8):
    """Graetz məsələsi: Theta = sum C_n R_n(r) exp(-lam_n^2 zeta). r = r/R."""
    r0 = 1e-6
    rr = np.linspace(r0, 1, 600)

    def shoot(lam):
        f = lambda r, y: [y[1], -y[1] / r - lam**2 * (1 - r**2) * y[0]]
        s = solve_ivp(f, [r0, 1], [1.0, -lam**2 * r0 / 2], t_eval=rr, rtol=1e-9, atol=1e-11)
        return s.y[0]

    lams, grid = [], np.arange(1.0, 4 * n + 4, 0.25)
    vals = [shoot(g)[-1] for g in grid]
    for a, b, va, vb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if va * vb < 0:
            lams.append(brentq(lambda l: shoot(l)[-1], a, b, xtol=1e-9))
        if len(lams) == n:
            break
    modes = np.array([shoot(l) for l in lams])
    w = rr * (1 - rr**2)
    C = np.array([trapezoid(w * m, rr) / trapezoid(w * m * m, rr) for m in modes])
    return np.array(lams), C, rr, modes


def _blasius():
    eta = np.linspace(0, 10, 801)
    s = solve_ivp(lambda e, y: [y[1], y[2], -0.5 * y[0] * y[2]], [0, 10],
                  [0, 0, 0.33206], t_eval=eta, rtol=1e-10, atol=1e-12)
    return eta, s.y[0], s.y[1]


def _pohlhausen(eta, f, Pr):
    from scipy.integrate import cumulative_trapezoid as ct
    d = np.exp(-0.5 * Pr * ct(f, eta, initial=0))
    th = ct(d, eta, initial=0)
    return th / th[-1]


def _ostrach(Pr, eta_max=12.0):
    def ode(e, y):
        f, fp, fpp, th, thp = y
        return np.vstack([fp, fpp, -3 * f * fpp + 2 * fp**2 - th, thp, -3 * Pr * f * thp])

    bc = lambda a, b: np.array([a[0], a[1], a[3] - 1, b[1], b[3]])
    e = np.linspace(0, eta_max, 400)
    g = np.vstack([1 - (e + 1) * np.exp(-e), e * np.exp(-e), (1 - e) * np.exp(-e),
                   np.exp(-e), -np.exp(-e)])
    s = solve_bvp(ode, bc, e, g, tol=1e-6, max_nodes=50000)
    if not s.success:
        raise RuntimeError("Şaquli lövhə həlli yığılmadı (Pr çox kənardadır).")
    return e, s.sol(e)


# ---------------- Plotly kömək funksiyaları ----------------
def _base(fig, N, S, T, tmin, tmax, bar):
    fig.add_trace(go.Surface(x=N, y=np.zeros_like(N), z=S, surfacecolor=T, cmin=tmin, cmax=tmax,
                             colorscale="Inferno", colorbar=dict(title=bar)))


def _cones(fig, n_pos, s_pos, vel):
    fig.add_trace(go.Cone(x=n_pos, y=np.zeros_like(n_pos), z=s_pos,
                          u=np.zeros_like(vel), v=np.zeros_like(vel), w=vel,
                          anchor="tail", sizemode="scaled", sizeref=0.5,
                          colorscale="Blues", showscale=False, name="Sürət"))


def _layout(fig, title, s_name, n_name):
    fig.update_layout(title=title, height=560, margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(xaxis_title=n_name, yaxis_title="", zaxis_title=s_name,
                                 yaxis=dict(showticklabels=False), aspectmode="data",
                                 camera=dict(eye=dict(x=1.5, y=-1.9, z=0.9))))


def _wall(fig, kind, size, length):
    if kind == "pipe":  # arxa yarım silindr
        z = np.linspace(0, length, 2)
        ph = np.linspace(0, np.pi, 40)
        X = size * np.outer(np.ones_like(z), np.cos(ph))
        Y = size * np.outer(np.ones_like(z), np.sin(ph))
        Z = np.outer(z, np.ones_like(ph))
    else:  # lövhə (n=0 müstəvisi)
        y = np.linspace(0, size, 2)
        z = np.linspace(0, length, 2)
        Y, Z = np.meshgrid(y, z)
        X = np.zeros_like(Y)
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=np.zeros_like(X), opacity=0.25,
                             colorscale=[[0, "gray"], [1, "gray"]], showscale=False))


# ---------------- Əsas giriş nöqtəsi ----------------
def build(geometry, p, pr):
    """geometry: SUPPORTED-dən biri; p: giriş dəyərləri (dict); pr: props() nəticəsi.
    Qaytarır: (plotly figure, info dict)."""
    fig, info = go.Figure(), {}
    if geometry in (PIPE_TW, PIPE_QS):
        D, L, U = p["D"], p["L"], p["U"]
        R = D / 2
        Re = U * D / pr["nu"]
        info.update(Re=Re, regime="laminar" if Re < 2300 else "turbulent (bu model laminar həlldir!)")
        n = np.linspace(-1, 1, 41)
        z = np.linspace(0, L, 80)
        N, S = np.meshgrid(n * R, z)
        rs = np.abs(N) / R
        if geometry == PIPE_TW:
            lam, C, rr, modes = _graetz()
            zeta = pr["alpha"] * S / (2 * U * R**2)
            Th = sum(c * np.interp(rs, rr, m) * np.exp(-l**2 * zeta) for c, m, l in zip(C, modes, lam))
            T = p["T_w"] - Th * (p["T_w"] - p["T_in"])
            info["Pe"] = U * D / pr["alpha"]
        else:
            q = p["q"]
            Tm = p["T_in"] + 2 * q * S / (U * R * (pr["k"] / pr["alpha"]))  # rho*cp = k/alpha
            T = Tm + (q * R / pr["k"]) * (rs**2 - rs**4 / 4 - 7 / 24)
            info["T_wall_exit"] = float(T[-1, 0])
        _wall(fig, "pipe", R, L)
        _base(fig, N, S, T, T.min(), T.max(), "T")
        zs = np.linspace(0, L, 5)[1:-1]
        ns = np.linspace(-0.9, 0.9, 9) * R
        nn, zz = np.meshgrid(ns, zs)
        _cones(fig, nn.ravel(), zz.ravel(), (2 * U * (1 - (nn / R) ** 2)).ravel())
        _layout(fig, geometry, "z (m)", "r (m)")

    elif geometry == PLATE_FORCED:
        L, U, Pr = p["L"], p["U"], pr["Pr"]
        Re_L = U * L / pr["nu"]
        info.update(Re_L=Re_L, regime="laminar" if Re_L < 5e5 else "keçid/turbulent (model yalnız laminardır!)")
        eta, f, fp = _blasius()
        th = _pohlhausen(eta, f, Pr)
        x = np.linspace(L * 0.02, L, 70)
        delta = 5 * np.sqrt(pr["nu"] * L / U)
        y = np.linspace(0, 1.2 * delta, 50)
        S, N = np.meshgrid(x, y, indexing="ij")
        E = N * np.sqrt(U / (pr["nu"] * S))
        T = p["T_w"] + np.interp(E, eta, th, right=1) * (p["T_inf"] - p["T_w"])
        _wall(fig, "plate", L * 0.3, L)
        _base(fig, N, S, T, min(p["T_w"], p["T_inf"]), max(p["T_w"], p["T_inf"]), "T")
        xs = np.linspace(L * 0.2, L, 4)
        ys = np.linspace(0.05, 1.1, 8) * delta
        yy, xx = np.meshgrid(ys, xs)
        ee = yy * np.sqrt(U / (pr["nu"] * xx))
        _cones(fig, yy.ravel(), xx.ravel(), (U * np.interp(ee, eta, fp, right=1)).ravel())
        _layout(fig, geometry, "x (m)", "y (m)")

    elif geometry == PLATE_NAT:
        H, Pr = p["H"], pr["Pr"]
        dT = p["T_w"] - p["T_inf"]
        sgn, g = (1 if dT >= 0 else -1), 9.81
        eta, sol = _ostrach(Pr)
        f, fp, th = sol[0], sol[1], sol[3]
        x = np.linspace(H * 0.02, H, 70)
        Grx = g * pr["beta"] * abs(dT) * x**3 / pr["nu"] ** 2
        info.update(Ra_H=float(Grx[-1] * Pr), regime="laminar" if Grx[-1] * Pr < 1e9 else "turbulent (model laminardır!)",
                    Nu_H=float((4 / 3) * (Grx[-1] / 4) ** 0.25 * -sol[4][0]))
        scale = x / (Grx / 4) ** 0.25  # y = eta * scale
        ymax = 12 * scale[-1] * 0.6
        y = np.linspace(0, ymax, 60)
        S, N = np.meshgrid(x, y, indexing="ij")
        E = N / scale[:, None]
        T = p["T_inf"] + np.interp(E, eta, th, right=0) * dT
        _wall(fig, "plate", H * 0.2, H)
        _base(fig, N, S, T, min(p["T_w"], p["T_inf"]), max(p["T_w"], p["T_inf"]), "T")
        xs = np.linspace(H * 0.2, H, 4)
        sc = xs / (g * pr["beta"] * abs(dT) * xs**3 / pr["nu"] ** 2 / 4) ** 0.25
        ys = np.linspace(0.3, 6, 8)
        yy = ys[None, :] * sc[:, None]
        Gr_s = g * pr["beta"] * abs(dT) * xs**3 / pr["nu"] ** 2
        u = (2 * pr["nu"] / xs)[:, None] * np.sqrt(Gr_s)[:, None] * np.interp(ys, eta, fp)[None, :]
        xx = np.repeat(xs[:, None], len(ys), axis=1)
        _cones(fig, yy.ravel(), xx.ravel(), sgn * u.ravel())
        _layout(fig, geometry, "x (m)", "y (m)")
    else:
        raise ValueError("Bu həndəsə üçün 3D hələlik yoxdur.")
    return fig, info
