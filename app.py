from pathlib import Path

import numpy as np
import streamlit as st

import plots
from fluidcalc import FLUIDS
from hybrid import HybridModel

st.set_page_config(page_title="Mayelər üçün hibrid hesablama", layout="wide")

LABELS = {
    "pipe_CWT": "Boru daxili axın, sabit divar temperaturu",
    "pipe_CHF": "Boru daxili axın, sabit istilik axını",
    "plate_forced": "Düz lövhə üzərində məcburi axın",
    "cylinder_cross": "Silindr ətrafında eninə axın",
    "sphere_forced": "Kürə ətrafında axın",
    "vplate_natural": "Şaquli lövhə, təbii konveksiya",
    "hcyl_natural": "Üfüqi silindr, təbii konveksiya",
}
NEEDS_VELOCITY = plots.FORCED
NEEDS_LD = plots.PIPES
L_HELP = {
    "pipe_CWT": "Boru diametri D (m)",
    "pipe_CHF": "Boru diametri D (m)",
    "plate_forced": "Lövhənin axın istiqamətində uzunluğu L (m)",
    "cylinder_cross": "Silindrin diametri D (m)",
    "sphere_forced": "Kürənin diametri D (m)",
    "vplate_natural": "Lövhənin hündürlüyü L (m)",
    "hcyl_natural": "Silindrin diametri D (m)",
}


@st.cache_resource
def load_model():
    return HybridModel(str(Path(__file__).parent / "hybrid_bundle.joblib"))


@st.cache_data(show_spinner="Qrafik hesablanır...")
def run_sweep(_model, base_items, var, lo, hi):
    return plots.sweep_data(_model, dict(base_items), var, lo, hi, n=40)


@st.cache_data(show_spinner="3D səth hesablanır (bir neçə saniyə)...")
def run_surface(_model, base_items, xvar, yvar, xlim, ylim):
    return plots.surface_data(_model, dict(base_items), xvar, yvar, xlim, ylim, n=15)


model = load_model()

st.title("Mayelər üçün hibrid hesablama sistemi")
st.warning(
    "ML düzəlişi sintetik data ilə öyrədilib və real eksperimental datada hələ doğrulanmayıb. "
    "Fizika nəticəsi (düstur və həll addımları) ML-dən asılı deyil."
)

with st.sidebar:
    st.header("Giriş")
    geometry = st.selectbox("Həndəsə", list(LABELS), format_func=lambda g: LABELS[g])
    fluid = st.selectbox("Maye", list(FLUIDS))
    t_lo, t_hi = FLUIDS[fluid][2], FLUIDS[fluid][3]
    st.caption(f"Bu maye üçün temperatur diapazonu: {t_lo - 273.15:.0f} … {t_hi - 273.15:.0f} °C")
    T_ref_c = st.number_input("Axın temperaturu T_ref (°C)", value=57.0, step=1.0)
    T_wall_c = st.number_input("Divar temperaturu T_wall (°C)", value=72.0, step=1.0)
    L = st.number_input(L_HELP[geometry], value=0.02, min_value=1e-4, format="%.4f")
    velocity = None
    if geometry in NEEDS_VELOCITY:
        velocity = st.number_input("Sürət v (m/s)", value=1.5, min_value=1e-4)
    L_over_D = None
    if geometry in NEEDS_LD:
        L_over_D = st.number_input("L/D", value=80.0, min_value=1.0)
    P_bar = st.number_input("Təzyiq (bar)", value=float(FLUIDS[fluid][1]) / 1e5, min_value=0.01)
    if st.button("Hesabla", type="primary"):
        # nəticə session_state-də saxlanır ki, qrafik seçimləri dəyişəndə itməsin
        st.session_state["base"] = dict(
            geometry=geometry, fluid=fluid, T_ref=T_ref_c + 273.15, T_wall=T_wall_c + 273.15,
            L=L, velocity=velocity, L_over_D=L_over_D, P=P_bar * 1e5,
        )

base = st.session_state.get("base")
if base is None:
    st.info("Soldakı sahələri doldurub «Hesabla» düyməsini basın.")
    st.stop()

try:
    out = model.predict(base["geometry"], base["fluid"], T_ref=base["T_ref"], T_wall=base["T_wall"],
                        L=base["L"], velocity=base["velocity"], L_over_D=base["L_over_D"], P=base["P"])
except Exception as e:  # solve() yanlış giriş üçün xəta qaldırır
    st.error(f"Hesablama alınmadı: {e}")
    st.stop()

r = out["result"]
base_items = tuple(sorted(base.items()))
bounds = (FLUIDS[base["fluid"]][2], FLUIDS[base["fluid"]][3])
avail = plots.available_vars(base["geometry"])

tab_res, tab_sweep, tab_surf, tab_pipe = st.tabs(
    ["Nəticə", "Həssaslıq qrafiki", "3D səth", "Boru 3D (illüstrasiya)"]
)

with tab_res:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{r.nu_name} (fizika)", f"{out['Nu_physics']:.4g}")
    c2.metric(f"{r.nu_name} (yekun)", f"{out['Nu']:.4g}")
    c3.metric("h, W/(m²·K)", f"{out['h']:.4g}")
    if out["used_ml"]:
        lo, hi = out["interval"]
        st.success(f"ML düzəlişi tətbiq olundu. 90% interval: {lo:.4g} … {hi:.4g}")
    else:
        st.info("Yalnız fizika nəticəsi göstərilir (ML düzəlişi tətbiq olunmadı).")
    for w in out["warnings"]:
        st.warning(w)
    st.subheader("Addım-addım həll")
    st.markdown(r.to_markdown())

with tab_sweep:
    st.caption("Hər nöqtə fizika moduluna və ML düzəlişinə ayrıca hesablanır. Digər giriş "
               "dəyərləri soldakı kimi qalır.")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    var = c1.selectbox("X oxu", avail, format_func=plots.var_label, key="sw_var")
    d_lo, d_hi = plots.default_range(var, base, bounds)
    lo = c2.number_input("Min", value=float(d_lo), key=f"sw_lo_{var}")
    hi = c3.number_input("Max", value=float(d_hi), key=f"sw_hi_{var}")
    metric = c4.radio("Göstərici", ["Nu", "h"], horizontal=True, key="sw_metric")
    logy = st.checkbox("Loqarifmik Y oxu", key="sw_log")
    if hi <= lo:
        st.error("Max dəyəri Min-dən böyük olmalıdır.")
    else:
        data = run_sweep(model, base_items, var, lo, hi)
        st.plotly_chart(plots.sweep_figure(data, var, metric, logy, r.nu_name),
                        use_container_width=True)
        if data["failed"]:
            st.caption(f"{data['failed']} nöqtə hesablana bilmədi (diapazondan kənar) və qrafikdə boşdur.")
        if np.isnan(data["Nu_lo"]).any():
            st.caption("İnterval zolağı yalnız ML düzəlişinin tətbiq olunduğu nöqtələrdə göstərilir.")

with tab_surf:
    st.caption("İki giriş dəyişəninə görə yekun nəticənin səthi. Qırmızı nöqtə cari girişdir. "
               "Siçanla fırladın və böyüdün.")
    c1, c2, c3 = st.columns(3)
    xvar = c1.selectbox("X oxu", avail, index=0, format_func=plots.var_label, key="sf_x")
    y_opts = [v for v in avail if v != xvar]
    yvar = c2.selectbox("Y oxu", y_opts, index=0, format_func=plots.var_label, key="sf_y")
    zmetric = c3.radio("Z oxu", ["h", "Nu"], horizontal=True, key="sf_z")
    xl = plots.default_range(xvar, base, bounds)
    yl = plots.default_range(yvar, base, bounds)
    data = run_surface(model, base_items, xvar, yvar,
                       (float(xl[0]), float(xl[1])), (float(yl[0]), float(yl[1])))
    x0 = float(plots.to_display(xvar, base[xvar]))
    y0 = float(plots.to_display(yvar, base[yvar]))
    st.plotly_chart(
        plots.surface_figure(data, xvar, yvar, zmetric,
                             point=(x0, y0, out[zmetric]), nu_name=r.nu_name),
        use_container_width=True)
    if data["failed"]:
        st.caption(f"{data['failed']} nöqtə hesablana bilmədi (diapazondan kənar) və səthdə boşdur.")

with tab_pipe:
    if base["geometry"] != "pipe_CWT":
        st.info("Boru vizuallaşdırması yalnız «Boru daxili axın, sabit divar temperaturu» həndəsəsi üçündür.")
    else:
        s = plots.pipe_state(base, out)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Giriş T", f"{s['Tin'] - 273.15:.1f} °C")
        c2.metric("Çıxış T", f"{s['Tout'] - 273.15:.1f} °C")
        c3.metric("İstilik axını Q", f"{s['Q'] / 1000:.3g} kVt")
        c4.metric("Boru uzunluğu", f"{s['Lp']:.3g} m")
        dT_out = abs(s["Tout"] - s["Tin"])
        if dT_out > 0.5 * abs(s["Tw"] - s["Tin"]):
            st.warning(f"Maye boruda {dT_out:.1f} °C qızır/soyuyur. Xassələr T_ref-də "
                       "götürüldüyü üçün bu nəticə təxminidir.")
        st.plotly_chart(plots.pipe_figure_3d(s), use_container_width=True)
        st.plotly_chart(plots.pipe_figure_2d(s), use_container_width=True)
        rejim = "laminar (parabolik profil)" if s["laminar"] else "turbulent (1/7 qüvvət qanunu profili)"
        st.caption(
            f"Rejim: {rejim}, Re = {s['Re']:.3g}, u_max ≈ {s['umax']:.3g} m/s. "
            "**Bu CFD simulyasiyası deyil.** Rəng boru boyu orta temperaturu göstərir "
            "(enerji balansı, orta h ilə, T_ref giriş temperaturu qəbul olunub). Sağdakı qabıq "
            "çıxışdakı sürət profilinin formasıdır. Radius miqyası şişirdilib, real boru "
            "bu şəkildən xeyli nazikdir."
        )
