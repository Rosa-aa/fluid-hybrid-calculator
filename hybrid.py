import numpy as np, pandas as pd, joblib
import CoolProp.CoolProp as CP
from fluidcalc import solve, FLUIDS

class HybridModel:
    def __init__(self, path):
        self.b = joblib.load(path)

    def features(self, geometry, fluid, T_ref, T_wall, P, r, L_over_D=None):
        v = r.values
        cp_id = FLUIDS[fluid][0]
        mu_ref = CP.PropsSI("V", "T", T_ref, "P", P, cp_id)
        mu_w = CP.PropsSI("V", "T", T_wall, "P", P, cp_id)
        lg = lambda x: float(np.log10(x)) if x else np.nan
        raw = {"log_Re": lg(v.get("Re")), "log_Ra": lg(v.get("Ra")),
               "log_Pr": lg(v.get("Pr")), "log_L_over_D": lg(L_over_D),
               "dT_over_T": (T_wall - T_ref) / T_ref,
               "mu_ratio_ref_wall": mu_ref / mu_w}
        row = {c: (-99.0 if np.isnan(raw[c]) else raw[c]) for c in self.b["num_features"]}
        fc = self.b["fluid_class"].get(fluid)
        for c in self.b["feature_columns"]:
            if c.startswith("geometry_"):
                row[c] = float(c == "geometry_" + geometry)
            elif c.startswith("fluid_class_"):
                row[c] = float(c == "fluid_class_" + str(fc))
        X = pd.DataFrame([row])[self.b["feature_columns"]]
        return X, raw

    def in_domain(self, geometry, fluid, raw):
        b = self.b
        if geometry not in b["bounds"]:
            return False, f"'{geometry}' həndəsəsi ML təlimində yoxdur"
        if fluid not in b["fluid_class"]:
            return False, f"'{fluid}' mayesi ML təlimində yoxdur"
        for c, (lo, hi) in b["bounds"][geometry].items():
            x = raw[c]
            if not np.isnan(x) and not (lo <= x <= hi):
                return False, f"{c} = {x:.3g} təlim diapazonundan ({lo:.3g} … {hi:.3g}) kənardadır"
        return True, ""

    def predict(self, geometry, fluid, T_ref, T_wall, L, velocity=None,
                L_over_D=None, P=101325.0):
        r = solve(geometry, fluid, T_ref=T_ref, T_wall=T_wall, L=L,
                  velocity=velocity, L_over_D=L_over_D, P=P)
        out = {"result": r, "Nu_physics": r.Nu, "Nu": r.Nu, "h": r.h,
               "used_ml": False, "interval": None, "warnings": list(r.warnings)}
        X, raw = self.features(geometry, fluid, T_ref, T_wall, P, r, L_over_D)
        ok, why = self.in_domain(geometry, fluid, raw)
        if not ok:
            out["warnings"].append("ML düzəlişi tətbiq olunmadı, yalnız fizika nəticəsi: " + why)
            return out
        corr = 10 ** float(self.b["model"].predict(X)[0])
        q = self.b["q90_log10"][geometry.split("_")[0]]
        Nu = r.Nu * corr
        out.update(Nu=Nu, h=r.h * corr, used_ml=True,
                   interval=(Nu * 10 ** -q, Nu * 10 ** q))
        return out
