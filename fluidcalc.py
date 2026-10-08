"""
fluidcalc.py - physics module (no ML) for convective heat-transfer problems.

solve(...) returns a Result with Nu, h, regime, the correlation used,
a list of step-by-step solution steps and a list of validity warnings.

Units: SI (T in K, P in Pa, lengths in m, velocity in m/s).

Geometries (key -> meaning of `L`):
  pipe_CWT        L = inner diameter D        needs velocity, L_over_D
  pipe_CHF        L = inner diameter D        needs velocity, L_over_D
  plate_forced    L = plate length            needs velocity
  cylinder_cross  L = diameter D              needs velocity
  sphere_forced   L = diameter D              needs velocity
  vplate_natural  L = plate height            needs T_wall != T_ref
  hcyl_natural    L = diameter D              needs T_wall != T_ref

Usage:
    from fluidcalc import solve
    r = solve("pipe_CWT", "Water", T_ref=330, T_wall=345, L=0.02, velocity=1.5, L_over_D=80)
    print(r.to_text())
"""
from dataclasses import dataclass, field
import numpy as np
from CoolProp.CoolProp import PropsSI

G = 9.81

# name: (CoolProp id, default P [Pa], T_min [K], T_max [K], class)  - same ranges as the dataset
FLUIDS = {
    "Water":       ("Water",           2.0e5, 285, 385, "liquid"),
    "Ethanol":     ("Ethanol",         1.0e5, 285, 340, "liquid"),
    "Methanol":    ("Methanol",        1.0e5, 285, 325, "liquid"),
    "Toluene":     ("Toluene",         1.0e5, 285, 380, "liquid"),
    "n-Heptane":   ("n-Heptane",       1.0e5, 285, 350, "liquid"),
    "R134a":       ("R134a",           1.0e6, 265, 300, "liquid"),
    "Propane":     ("Propane",         2.0e6, 260, 320, "liquid"),
    "Ammonia":     ("Ammonia",         2.0e6, 260, 320, "liquid"),
    "Air":         ("Air",             1.0e5, 260, 600, "gas"),
    "Nitrogen":    ("Nitrogen",        1.0e5, 260, 600, "gas"),
    "Argon":       ("Argon",           1.0e5, 260, 600, "gas"),
    "Helium":      ("Helium",          1.0e5, 260, 600, "gas"),
    "CO2":         ("CO2",             1.0e5, 280, 600, "gas"),
    "Hydrogen":    ("Hydrogen",        1.0e5, 260, 600, "gas"),
    "MEG30":       ("INCOMP::MEG-30%", 1.0e5, 270, 370, "incompressible_liquid"),
    "DowQ":        ("INCOMP::DowQ",    1.0e5, 270, 500, "incompressible_liquid"),
    "Therminol66": ("INCOMP::T66",     1.0e5, 300, 500, "incompressible_liquid"),
}

GEOMETRIES = {
    "pipe_CWT":       "Boru daxili axın, sabit divar temperaturu (L = diametr D)",
    "pipe_CHF":       "Boru daxili axın, sabit istilik axını (L = diametr D)",
    "plate_forced":   "Düz lövhə üzərində məcburi axın (L = lövhənin uzunluğu)",
    "cylinder_cross": "Silindr ətrafında eninə axın (L = diametr D)",
    "sphere_forced":  "Kürə ətrafında axın (L = diametr D)",
    "vplate_natural": "Şaquli lövhə, təbii konveksiya (L = hündürlük)",
    "hcyl_natural":   "Üfüqi silindr, təbii konveksiya (L = diametr D)",
}


# ----------------------------------------------------------------------------- result types
@dataclass
class Step:
    title: str
    formula: str = ""
    calc: str = ""
    result: str = ""


@dataclass
class Result:
    geometry: str
    fluid: str
    Nu: float
    h: float
    nu_name: str                    # "Nu_D" or "Nu_L"
    regime: str
    correlation: str
    steps: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    values: dict = field(default_factory=dict)   # Re, Pr, Ra, rho, mu, k, cp ... for later ML use

    def to_text(self):
        out = [f"Məsələ: {GEOMETRIES[self.geometry]} | Maye: {self.fluid}", ""]
        for i, s in enumerate(self.steps, 1):
            out.append(f"{i}. {s.title}")
            for label, txt in (("düstur", s.formula), ("əvəzetmə", s.calc), ("nəticə", s.result)):
                if txt:
                    out.append(f"   {label}: {txt}")
        out += ["", f"NƏTİCƏ: {self.nu_name} = {_f(self.Nu)},  h = {_f(self.h)} W/(m²·K)",
                f"Rejim: {self.regime} | Düstur: {self.correlation}"]
        if self.warnings:
            out += ["", "Xəbərdarlıqlar:"] + [f" - {w}" for w in self.warnings]
        return "\n".join(out)

    def to_markdown(self):
        out = [f"**Məsələ:** {GEOMETRIES[self.geometry]} · **Maye:** {self.fluid}", ""]
        for i, s in enumerate(self.steps, 1):
            out.append(f"**{i}. {s.title}**")
            if s.formula:
                out.append(f"- Düstur: `{s.formula}`")
            if s.calc:
                out.append(f"- Əvəzetmə: `{s.calc}`")
            if s.result:
                out.append(f"- Nəticə: **{s.result}**")
            out.append("")
        out.append(f"### {self.nu_name} = {_f(self.Nu)}  ·  h = {_f(self.h)} W/(m²·K)")
        out.append(f"Rejim: {self.regime} · Düstur: {self.correlation}")
        if self.warnings:
            out += ["", "**Xəbərdarlıqlar**"] + [f"- {w}" for w in self.warnings]
        return "\n".join(out)


def _f(x, n=4):
    return f"{x:.{n}g}"


# ----------------------------------------------------------------------------- properties
def _props(cp_id, T, P):
    try:
        rho = PropsSI("D", "T", T, "P", P, cp_id)
        mu = PropsSI("V", "T", T, "P", P, cp_id)
        k = PropsSI("L", "T", T, "P", P, cp_id)
        cp = PropsSI("C", "T", T, "P", P, cp_id)
    except ValueError as e:
        raise ValueError(f"T = {T:.1f} K və P = {P/1e5:.2f} bar üçün xassələr hesablana bilmədi "
                         f"(bu maye üçün tək fazalı oblastdan kənardır). Detal: {e}") from None
    return dict(rho=rho, mu=mu, k=k, cp=cp, Pr=cp * mu / k)


def _beta(cp_id, T, P):
    d = 0.5
    r1 = PropsSI("D", "T", T - d, "P", P, cp_id)
    r2 = PropsSI("D", "T", T + d, "P", P, cp_id)
    r0 = PropsSI("D", "T", T, "P", P, cp_id)
    return -(r2 - r1) / (2 * d) / r0


# ----------------------------------------------------------------------------- correlations
def nu_lam_hausen(Re, Pr, D_over_L):
    Gz = D_over_L * Re * Pr
    return 3.66 + 0.0668 * Gz / (1.0 + 0.04 * Gz ** (2.0 / 3.0))


def nu_lam_chf_mean(Re, Pr, L_over_D):
    X = L_over_D / (Re * Pr)
    a, b = 5e-5, 1e-3
    seg1 = lambda u: 1.302 * 1.5 * u ** (2 / 3) - u
    seg2 = lambda lo, hi: 1.302 * 1.5 * (hi ** (2 / 3) - lo ** (2 / 3)) - 0.5 * (hi - lo)
    if X <= a:
        I = seg1(X)
    elif X <= b:
        I = seg1(a) + seg2(a, X)
    else:
        s = np.geomspace(b, X, 200)
        f = 4.364 + 8.68 * (1e3 * s) ** -0.506 * np.exp(-41.0 * s)
        I = seg1(a) + seg2(a, b) + np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(s))
    return I / X, X


def gnielinski(Re, Pr, D_over_L):
    f = (0.79 * np.log(Re) - 1.64) ** -2
    num = (f / 8.0) * (Re - 1000.0) * Pr
    den = 1.0 + 12.7 * np.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0)
    return (num / den) * (1.0 + D_over_L ** (2.0 / 3.0)), f


# ----------------------------------------------------------------------------- solvers
def _solve_pipe(bc, fl, T, Tw, D, v, LD, P, st, warn, vals):
    cp_id = fl[0]
    pb, pw = _props(cp_id, T, P), _props(cp_id, Tw, P)
    st.append(Step("Mayenin xassələri (orta axın temperaturunda)",
                   "ρ, μ, k, cp - CoolProp; Pr = cp·μ/k",
                   f"T = {_f(T)} K, P = {_f(P/1e5)} bar",
                   f"ρ = {_f(pb['rho'])} kg/m³, μ = {_f(pb['mu'])} Pa·s, k = {_f(pb['k'])} W/(m·K), "
                   f"cp = {_f(pb['cp'])} J/(kg·K)"))
    Re = pb["rho"] * v * D / pb["mu"]
    Pr, Prw = pb["Pr"], pw["Pr"]
    st.append(Step("Reynolds ədədi", "Re = ρ·v·D/μ",
                   f"Re = {_f(pb['rho'])}·{_f(v)}·{_f(D)}/{_f(pb['mu'])}", f"Re = {_f(Re)}"))
    st.append(Step("Prandtl ədədi", "Pr = cp·μ/k",
                   f"Pr = {_f(pb['cp'])}·{_f(pb['mu'])}/{_f(pb['k'])}",
                   f"Pr = {_f(Pr)}  (divar temperaturunda Pr_w = {_f(Prw)})"))
    D_over_L = 1.0 / LD
    corr_f = (Pr / Prw) ** 0.11
    if not (0.5 <= Pr <= 2000):
        warn.append(f"Pr = {_f(Pr)} düsturun etibarlılıq diapazonundan (0.5-2000) kənardadır.")

    def lam(R):
        if bc == "CWT":
            return nu_lam_hausen(R, Pr, D_over_L)
        return nu_lam_chf_mean(R, Pr, LD)[0]

    vals.update(Re=Re, Pr=Pr, Pr_wall=Prw, rho=pb["rho"], mu=pb["mu"], k=pb["k"], cp=pb["cp"])
    if Re < 2300:
        regime = "laminar"
        st.append(Step("Axın rejimi", "Re < 2300 → laminar", f"Re = {_f(Re)}", "laminar"))
        if bc == "CWT":
            Gz = D_over_L * Re * Pr
            Nu = nu_lam_hausen(Re, Pr, D_over_L)
            st.append(Step("Graetz ədədi", "Gz = (D/L)·Re·Pr",
                           f"Gz = (1/{_f(LD)})·{_f(Re)}·{_f(Pr)}", f"Gz = {_f(Gz)}"))
            st.append(Step("Hausen düsturu (termik inkişaf, orta Nu)",
                           "Nu = 3.66 + 0.0668·Gz / (1 + 0.04·Gz^(2/3))",
                           f"Nu = 3.66 + 0.0668·{_f(Gz)} / (1 + 0.04·{_f(Gz)}^(2/3))", f"Nu = {_f(Nu)}"))
            corr = "Hausen (laminar, sabit divar temperaturu)"
        else:
            Nu, X = nu_lam_chf_mean(Re, Pr, LD)
            st.append(Step("Ölçüsüz məsafə", "x* = (L/D)/(Re·Pr)",
                           f"x* = {_f(LD)}/({_f(Re)}·{_f(Pr)})", f"x* = {_f(X)}"))
            st.append(Step("Shah-London düsturu (sabit istilik axını)",
                           "lokal Nu(x*) boru uzunluğu boyunca orta qiymətlənir; "
                           "x* böyük olduqda Nu → 4.364",
                           f"x* = {_f(X)}", f"Nu = {_f(Nu)}"))
            corr = "Shah-London (laminar, sabit istilik axını), orta Nu"
    elif Re < 4000:
        regime = "keçid"
        g = (Re - 2300.0) / 1700.0
        nl, nt = lam(2300.0), gnielinski(4000.0, Pr, D_over_L)[0] * corr_f
        Nu = (1 - g) * nl + g * nt
        st.append(Step("Axın rejimi", "2300 ≤ Re < 4000 → keçid", f"Re = {_f(Re)}", "keçid"))
        st.append(Step("Laminar və turbulent qiymətlərin interpolyasiyası",
                       "Nu = (1-γ)·Nu_lam(2300) + γ·Nu_turb(4000), γ = (Re-2300)/1700",
                       f"γ = {_f(g)}, Nu_lam = {_f(nl)}, Nu_turb = {_f(nt)}", f"Nu = {_f(Nu)}"))
        warn.append("Keçid rejimində qeyri-müəyyənlik yüksəkdir (təxminən ±12% və daha çox).")
        corr = "Laminar-turbulent interpolyasiya (Gnielinski tövsiyəsi)"
    else:
        regime = "turbulent"
        Ng, f = gnielinski(Re, Pr, D_over_L)
        Nu = Ng * corr_f
        st.append(Step("Axın rejimi", "Re ≥ 4000 → turbulent", f"Re = {_f(Re)}", "turbulent"))
        st.append(Step("Sürtünmə əmsalı (Petukhov)", "f = (0.79·ln Re − 1.64)^(-2)",
                       f"f = (0.79·ln({_f(Re)}) − 1.64)^(-2)", f"f = {_f(f)}"))
        st.append(Step("Gnielinski düsturu (giriş effekti ilə)",
                       "Nu = (f/8)(Re−1000)Pr / (1+12.7√(f/8)(Pr^(2/3)−1)) · [1+(D/L)^(2/3)]",
                       f"f/8 = {_f(f/8)}, D/L = {_f(D_over_L)}", f"Nu_Gnielinski = {_f(Ng)}"))
        st.append(Step("Xassələrin dəyişməsi üçün düzəliş", "(Pr/Pr_w)^0.11",
                       f"({_f(Pr)}/{_f(Prw)})^0.11", f"düzəliş = {_f(corr_f)} → Nu = {_f(Nu)}"))
        if not (0.5 <= Pr / Prw <= 2):
            warn.append("Pr/Pr_w diapazondan (0.5-2) kənardadır, xassə düzəlişi etibarsız ola bilər.")
        corr = "Gnielinski + (Pr/Pr_w)^0.11"
        if Re > 5e6:
            warn.append("Re > 5·10⁶: Gnielinski düsturunun diapazonundan kənar.")
    if LD < 10 and regime != "laminar":
        warn.append("L/D < 10: giriş effekti düsturun dəqiqliyini azalda bilər.")
    return Nu, regime, corr


def _solve_external(geom, fl, T, Tw, L, v, P, st, warn, vals):
    cp_id = fl[0]
    pw = _props(cp_id, Tw, P)
    if geom == "sphere_forced":
        Tp, tag = T, "sərbəst axın temperaturunda"
    else:
        Tp, tag = 0.5 * (T + Tw), "film temperaturunda T_f = (T∞+T_w)/2"
    p = _props(cp_id, Tp, P)
    st.append(Step(f"Mayenin xassələri ({tag})", "ρ, μ, k, cp - CoolProp; Pr = cp·μ/k",
                   f"T = {_f(Tp)} K, P = {_f(P/1e5)} bar",
                   f"ρ = {_f(p['rho'])}, μ = {_f(p['mu'])}, k = {_f(p['k'])}, cp = {_f(p['cp'])}, "
                   f"Pr = {_f(p['Pr'])}"))
    Re = p["rho"] * v * L / p["mu"]
    Pr = p["Pr"]
    st.append(Step("Reynolds ədədi", "Re = ρ·v·L/μ",
                   f"Re = {_f(p['rho'])}·{_f(v)}·{_f(L)}/{_f(p['mu'])}", f"Re = {_f(Re)}"))
    vals.update(Re=Re, Pr=Pr, Pr_wall=pw["Pr"], rho=p["rho"], mu=p["mu"], k=p["k"], cp=p["cp"])

    if geom == "plate_forced":
        if not (0.6 <= Pr <= 60):
            warn.append(f"Pr = {_f(Pr)} lövhə düsturunun diapazonundan (0.6-60) kənardadır.")
        if Re < 5e5:
            Nu, regime = 0.664 * Re ** 0.5 * Pr ** (1 / 3), "laminar"
            st.append(Step("Rejim: laminar sərhəd layı", "Re_L < 5·10⁵", f"Re = {_f(Re)}", "laminar"))
            st.append(Step("Orta Nu (laminar lövhə)", "Nu_L = 0.664·Re^0.5·Pr^(1/3)",
                           f"Nu = 0.664·{_f(Re)}^0.5·{_f(Pr)}^(1/3)", f"Nu = {_f(Nu)}"))
            corr = "0.664·Re^0.5·Pr^(1/3) (laminar lövhə)"
        else:
            Nu, regime = (0.037 * Re ** 0.8 - 871.0) * Pr ** (1 / 3), "qarışıq (laminar+turbulent)"
            st.append(Step("Rejim: qarışıq sərhəd layı", "Re_L ≥ 5·10⁵", f"Re = {_f(Re)}", "qarışıq"))
            st.append(Step("Orta Nu (qarışıq sərhəd layı)", "Nu_L = (0.037·Re^0.8 − 871)·Pr^(1/3)",
                           f"Nu = (0.037·{_f(Re)}^0.8 − 871)·{_f(Pr)}^(1/3)", f"Nu = {_f(Nu)}"))
            corr = "(0.037·Re^0.8 − 871)·Pr^(1/3) (qarışıq lövhə)"
            if Re > 1e8:
                warn.append("Re > 10⁸: düstur diapazondan kənardadır.")
    elif geom == "cylinder_cross":
        Nu = 0.3 + (0.62 * Re ** 0.5 * Pr ** (1 / 3) / (1 + (0.4 / Pr) ** (2 / 3)) ** 0.25) * \
            (1 + (Re / 282000.0) ** (5 / 8)) ** (4 / 5)
        regime = "xarici məcburi axın"
        st.append(Step("Churchill-Bernstein düsturu (orta Nu_D)",
                       "Nu = 0.3 + [0.62Re^½Pr^⅓ / (1+(0.4/Pr)^⅔)^¼]·[1+(Re/282000)^⅝]^⅘",
                       f"Re = {_f(Re)}, Pr = {_f(Pr)}", f"Nu = {_f(Nu)}"))
        corr = "Churchill-Bernstein"
        if Re * Pr <= 0.2:
            warn.append("Re·Pr ≤ 0.2: düsturun diapazonundan kənar.")
    else:
        mr = p["mu"] / pw["mu"]
        if not (3.5 <= Re <= 7.6e4):
            warn.append(f"Re = {_f(Re)} Whitaker diapazonundan (3.5-7.6·10⁴) kənardadır.")
        if not (0.71 <= Pr <= 380):
            warn.append(f"Pr = {_f(Pr)} Whitaker diapazonundan (0.71-380) kənardadır.")
        if not (1.0 <= mr <= 3.2):
            warn.append(f"μ∞/μ_s = {_f(mr)} nominal diapazondan (1-3.2) kənardadır (ekstrapolyasiya).")
        Nu = 2.0 + (0.4 * Re ** 0.5 + 0.06 * Re ** (2 / 3)) * Pr ** 0.4 * mr ** 0.25
        regime = "xarici məcburi axın"
        st.append(Step("Whitaker düsturu (orta Nu_D)",
                       "Nu = 2 + (0.4Re^½ + 0.06Re^⅔)·Pr^0.4·(μ∞/μ_s)^¼",
                       f"μ∞/μ_s = {_f(mr)}, Re = {_f(Re)}, Pr = {_f(Pr)}", f"Nu = {_f(Nu)}"))
        corr = "Whitaker"
    return Nu, regime, corr


def _solve_natural(geom, fl, T, Tw, L, P, st, warn, vals):
    cp_id = fl[0]
    Tf = 0.5 * (T + Tw)
    p = _props(cp_id, Tf, P)
    beta = _beta(cp_id, Tf, P)
    dT = abs(Tw - T)
    st.append(Step("Mayenin xassələri (film temperaturunda T_f = (T∞+T_w)/2)",
                   "ρ, μ, k, cp - CoolProp; β = −(1/ρ)(∂ρ/∂T)_P",
                   f"T_f = {_f(Tf)} K, P = {_f(P/1e5)} bar",
                   f"ρ = {_f(p['rho'])}, μ = {_f(p['mu'])}, k = {_f(p['k'])}, cp = {_f(p['cp'])}, "
                   f"Pr = {_f(p['Pr'])}, β = {_f(beta)} 1/K"))
    Ra = G * beta * dT * L ** 3 * p["Pr"] * (p["rho"] / p["mu"]) ** 2
    st.append(Step("Rayleigh ədədi", "Ra = g·β·ΔT·L³·Pr/ν²,  ν = μ/ρ",
                   f"Ra = 9.81·{_f(beta)}·{_f(dT)}·{_f(L)}³·{_f(p['Pr'])}/({_f(p['mu']/p['rho'])})²",
                   f"Ra = {_f(Ra)}"))
    vals.update(Ra=Ra, Pr=p["Pr"], beta=beta, rho=p["rho"], mu=p["mu"], k=p["k"], cp=p["cp"])
    if beta <= 0:
        warn.append("β ≤ 0: bu temperaturda təbii konveksiya düsturu tətbiq olunmur (məs., suyun 4 °C ətrafı).")
    if not (1e3 <= Ra <= 1e12):
        warn.append(f"Ra = {_f(Ra)} təlim diapazonundan (10³-10¹²) kənardadır.")
    if geom == "vplate_natural":
        Nu = (0.825 + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / p["Pr"]) ** (9 / 16)) ** (8 / 27)) ** 2
        st.append(Step("Churchill-Chu düsturu (şaquli lövhə, orta Nu_L)",
                       "Nu = {0.825 + 0.387Ra^⅙ / [1+(0.492/Pr)^(9/16)]^(8/27)}²",
                       f"Ra = {_f(Ra)}, Pr = {_f(p['Pr'])}", f"Nu = {_f(Nu)}"))
        corr = "Churchill-Chu (şaquli lövhə)"
    else:
        Nu = (0.6 + 0.387 * Ra ** (1 / 6) / (1 + (0.559 / p["Pr"]) ** (9 / 16)) ** (8 / 27)) ** 2
        st.append(Step("Churchill-Chu düsturu (üfüqi silindr, orta Nu_D)",
                       "Nu = {0.6 + 0.387Ra^⅙ / [1+(0.559/Pr)^(9/16)]^(8/27)}²",
                       f"Ra = {_f(Ra)}, Pr = {_f(p['Pr'])}", f"Nu = {_f(Nu)}"))
        corr = "Churchill-Chu (üfüqi silindr)"
    return Nu, "təbii konveksiya", corr


# ----------------------------------------------------------------------------- public API
def solve(geometry, fluid, T_ref, T_wall, L, velocity=None, L_over_D=None, P=None):
    """Solve a convective heat-transfer problem and return a Result with step-by-step text."""
    if geometry not in GEOMETRIES:
        raise ValueError(f"Naməlum həndəsə: {geometry}. Mümkün: {', '.join(GEOMETRIES)}")
    if fluid not in FLUIDS:
        raise ValueError(f"Naməlum maye: {fluid}. Mümkün: {', '.join(FLUIDS)}")
    fl = FLUIDS[fluid]
    P = fl[1] if P is None else P
    if L <= 0:
        raise ValueError("L müsbət olmalıdır.")
    natural = geometry.endswith("natural")
    if not natural and (velocity is None or velocity <= 0):
        raise ValueError("Bu həndəsə üçün müsbət sürət (velocity) lazımdır.")
    if natural and abs(T_wall - T_ref) < 1e-6:
        raise ValueError("Təbii konveksiya üçün T_wall və T_ref fərqli olmalıdır.")
    if geometry.startswith("pipe") and (L_over_D is None or L_over_D <= 0):
        raise ValueError("Boru üçün müsbət L_over_D lazımdır.")

    warn, st, vals = [], [], {}
    for name, T in (("T_ref", T_ref), ("T_wall", T_wall)):
        if not (fl[2] <= T <= fl[3]):
            warn.append(f"{name} = {T:.1f} K bu maye üçün qəbul edilmiş diapazondan "
                        f"({fl[2]}-{fl[3]} K) kənardadır.")
    inp = f"həndəsə = {geometry}, maye = {fluid}, T_ref = {T_ref} K, T_wall = {T_wall} K, L = {L} m"
    if velocity is not None:
        inp += f", v = {velocity} m/s"
    if L_over_D is not None:
        inp += f", L/D = {L_over_D}"
    st.append(Step("Giriş məlumatları", "", inp, ""))

    if geometry.startswith("pipe"):
        bc = geometry.split("_")[1]
        Nu, regime, corr = _solve_pipe(bc, fl, T_ref, T_wall, L, velocity, L_over_D, P, st, warn, vals)
    elif natural:
        Nu, regime, corr = _solve_natural(geometry, fl, T_ref, T_wall, L, P, st, warn, vals)
    else:
        Nu, regime, corr = _solve_external(geometry, fl, T_ref, T_wall, L, velocity, P, st, warn, vals)

    nu_name = "Nu_L" if geometry in ("plate_forced", "vplate_natural") else "Nu_D"
    h = Nu * vals["k"] / L
    st.append(Step("İstilikötürmə əmsalı", f"h = {nu_name}·k/L", f"h = {_f(Nu)}·{_f(vals['k'])}/{_f(L)}",
                   f"h = {_f(h)} W/(m²·K)"))
    return Result(geometry, fluid, float(Nu), float(h), nu_name, regime, corr, st, warn, vals)


if __name__ == "__main__":
    print(solve("pipe_CWT", "Water", T_ref=330, T_wall=345, L=0.02, velocity=1.5, L_over_D=80).to_text())
