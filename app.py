from pathlib import Path

import streamlit as st

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
NEEDS_VELOCITY = {"pipe_CWT", "pipe_CHF", "plate_forced", "cylinder_cross", "sphere_forced"}
NEEDS_LD = {"pipe_CWT", "pipe_CHF"}
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
    run = st.button("Hesabla", type="primary")

if run:
    try:
        out = model.predict(
            geometry, fluid,
            T_ref=T_ref_c + 273.15, T_wall=T_wall_c + 273.15, L=L,
            velocity=velocity, L_over_D=L_over_D, P=P_bar * 1e5,
        )
    except Exception as e:  # solve() yanlış giriş üçün xəta qaldırır
        st.error(f"Hesablama alınmadı: {e}")
        st.stop()

    r = out["result"]
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
else:
    st.info("Soldakı sahələri doldurub «Hesabla» düyməsini basın.")
