# Hybrid Heat-Transfer Calculator for Fluids

A calculation system that combines **classical heat-transfer correlations** with a **machine-learning correction layer**. Every result is delivered with the formula, the substitution and a **step-by-step solution**; the ML correction comes with a calibrated **90% prediction interval** and an **applicability-domain check** that switches the ML off when the inputs fall outside what it was trained on.

**Live demo:** https://fluid-hybrid-calculator-zvllslakbaalr7tbxeed3j.streamlit.app/

> **Important note on scope.** The ML correction layer was trained on a **synthetic dataset** generated from established correlations and has **not yet been validated against real experimental measurements**. The physics part of the system (fluid properties, correlations, worked solutions) does not depend on the ML and was verified independently. See [Limitations](#limitations) for details. Validating against real data is the main next step.

---

## Table of contents

1. [What the project does](#what-the-project-does)
2. [Quick start](#quick-start)
3. [How to use the app](#how-to-use-the-app)
4. [Background: what is being computed](#background-what-is-being-computed)
5. [System design](#system-design)
6. [Physics module](#physics-module)
7. [Dataset](#dataset)
8. [Hybrid ML model](#hybrid-ml-model)
9. [Uncertainty and applicability domain](#uncertainty-and-applicability-domain)
10. [Experiments and results](#experiments-and-results)
11. [Verification and worked examples](#verification-and-worked-examples)
12. [Visualizations](#visualizations)
13. [Repository structure](#repository-structure)
14. [Running locally and deployment](#running-locally-and-deployment)
15. [Limitations](#limitations)
16. [Roadmap](#roadmap)
17. [References](#references)
18. [Development notes](#development-notes)
19. [Author](#author)

---

## What the project does

Engineers routinely need the heat-transfer coefficient *h* for a fluid flowing in a pipe, around a cylinder, or next to a heated plate. The usual workflow is: look up fluid properties, compute Reynolds and Prandtl numbers, decide the flow regime, choose a correlation, substitute, and compute. This tool automates that chain and **shows every step**, so it can be used for both calculation and learning.

On top of the classical result it adds a data-driven correction and quantifies how far that correction can be trusted.

**Key features**

- **7 flow configurations**
  - Internal flow in a pipe, constant wall temperature
  - Internal flow in a pipe, constant heat flux
  - Forced flow over a flat plate
  - Cross-flow over a cylinder
  - Forced flow around a sphere
  - Natural convection on a vertical plate
  - Natural convection on a horizontal cylinder
- **17 fluids** with properties pulled automatically from [CoolProp](http://www.coolprop.org/): 8 liquids (water, ethanol, methanol, toluene, n-heptane, R134a, propane, ammonia), 6 gases (air, nitrogen, argon, helium, CO₂, hydrogen) and 3 glycol/oil fluids (30% MEG, Dowtherm Q, Therminol 66). The user never types density, viscosity or conductivity.
- **Step-by-step solution** in Markdown: formula, substitution, numerical result for each step, including regime selection and correlation validity warnings.
- **Hybrid correction**: `Nu = Nu_physics × correction`, where the correction is learned by a gradient-boosted model.
- **Calibrated uncertainty**: a split-conformal prediction interval, `Nu ± x%`, with verified 90% coverage on the synthetic test sets.
- **Applicability-domain guard**: if the geometry, fluid or any input is outside the training range, the app returns the physics result only and says why.
- **Interactive charts**: sensitivity curves with the uncertainty band, a 3D response surface, and a 3D illustration of the temperature development along a pipe.

---

## Quick start

1. Open the [live demo](https://fluid-hybrid-calculator-zvllslakbaalr7tbxeed3j.streamlit.app/).
2. In the sidebar choose **Pipe, constant wall temperature**, fluid **Water**, and enter:

   | Field | Value |
   |---|---|
   | Bulk temperature `T_ref` | 57 °C |
   | Wall temperature `T_wall` | 72 °C |
   | Pipe diameter `D` | 0.02 m |
   | Velocity `v` | 1.5 m/s |
   | `L/D` | 80 |
   | Pressure | 2 bar |

3. Press **Calculate**. Expected: `Re ≈ 6.04·10⁴` (turbulent), `Nu ≈ 293.8`, `h ≈ 9519 W/(m²·K)` for the physics result. The final value with the ML correction differs by a fraction of a percent.

---

## How to use the app

The sidebar collects the inputs. Only the fields relevant to the chosen geometry are shown (for example, velocity is hidden for natural convection and `L/D` appears only for pipes).

| Input | Unit | Notes |
|---|---|---|
| Temperatures `T_ref`, `T_wall` | °C | Converted to kelvin internally |
| Characteristic length | m | Pipe/cylinder/sphere diameter, plate length or height |
| Velocity | m/s | Forced-flow geometries only |
| `L/D` | – | Pipes only |
| Pressure | bar | Converted to pascal internally |

**On pressure.** For liquids, pressure barely changes the result because they are nearly incompressible; the default per fluid is fine. For gases, pressure matters strongly because density scales with pressure and therefore changes the Reynolds (or Rayleigh) number. Enter the real operating pressure for gases.

After pressing **Calculate** the results are organised into four tabs:

| Tab | Content |
|---|---|
| **Result** | Physics `Nu`, final `Nu` (physics + ML), `h`, the 90% interval, warnings, a table of fluid properties and dimensionless numbers with units, and the full step-by-step solution |
| **Sensitivity** | `Nu` or `h` versus one input (velocity, temperature, size, `L/D`), physics and final curves, with the ML interval as a shaded band |
| **3D surface** | `h` or `Nu` as a surface over two chosen inputs; the red marker is the current input |
| **Pipe 3D (illustration)** | Bulk temperature along the pipe and the velocity-profile shape at the outlet (constant-wall-temperature pipe only) |

---

## Background: what is being computed

- **Nusselt number** `Nu = h·L / k` is the dimensionless heat-transfer coefficient: the ratio of convective to purely conductive heat transfer across the fluid layer.
- **Heat-transfer coefficient** `h` (W/(m²·K)) follows from `h = Nu·k / L`.
- **Reynolds number** `Re = ρ·v·D / μ` measures the ratio of inertial to viscous forces and decides whether the flow is laminar or turbulent.
- **Prandtl number** `Pr = c_p·μ / k` compares momentum diffusion with thermal diffusion; it depends only on the fluid.
- **Rayleigh number** `Ra = g·β·ΔT·L³·Pr / ν²` (with `ν = μ/ρ`) governs buoyancy-driven natural convection.

Correlations express `Nu` as a function of these groups, for example `Nu = f(Re, Pr)` in forced convection and `Nu = f(Ra, Pr)` in natural convection. Because they are empirical, each correlation has a stated validity range, and the physics module warns when the inputs leave it.

---

## System design

```
 user inputs
     │
     ▼
 fluidcalc.py ── CoolProp properties ─► Re, Pr / Ra ─► regime ─► correlation ─► Nu_physics, h
     │                                                             └─► step-by-step solution
     ▼
 hybrid.py ──► applicability-domain check ──┬─ inside  ─► XGBoost correction ─► Nu_final ± interval
                                            └─ outside ─► physics result + warning
     │
     ▼
 app.py (Streamlit)  +  plots.py (Plotly)
```

The design principle is **physics first, ML as a bounded correction**. The ML model never replaces the physics: it predicts a multiplicative correction around the physics result, and it is bypassed when the situation is unfamiliar.

---

## Physics module

`fluidcalc.py` is self-contained and works without any ML.

**Fluid properties.** Density, viscosity, conductivity and specific heat come from CoolProp at the relevant temperature: bulk temperature for forced flow, film temperature `(T_ref + T_wall)/2` for natural convection. The thermal-expansion coefficient `β` is obtained from the density derivative. The Prandtl number at the wall temperature is used for the property-variation correction.

**Regime selection.** Laminar for `Re < 2300`, turbulent for `Re ≥ 4000`, with a transitional range that is bridged so that `Nu` is continuous across both boundaries (tested at `Re = 2300` and `Re = 4000`).

**Correlations used**

- **Internal flow, laminar:** Hausen correlation for a developing flow at constant wall temperature (limit 3.66 for fully developed flow); fully developed limit 4.364 for constant heat flux.
- **Internal flow, turbulent:** Gnielinski correlation with the Petukhov friction factor, an entrance-length factor and a property-variation correction:

  ```
  f  = (0.79·ln(Re) − 1.64)^(−2)
  Nu = (f/8)(Re − 1000)Pr / (1 + 12.7·√(f/8)·(Pr^(2/3) − 1)) · [1 + (D/L)^(2/3)] · (Pr/Pr_w)^0.11
  ```

- **External flow:** Churchill-Bernstein (cylinder in cross-flow), Whitaker (sphere), standard laminar/turbulent flat-plate correlations.
- **Natural convection:** Churchill-Chu for the vertical plate:

  ```
  Nu = { 0.825 + 0.387·Ra^(1/6) / [1 + (0.492/Pr)^(9/16)]^(8/27) }²
  ```

  and a corresponding correlation for the horizontal cylinder. The exact formula used is always printed in the step-by-step output.

**Validity checks.** Each correlation carries its stated range (for example Prandtl-number limits). Inputs outside the range produce a warning; for dataset generation, out-of-range samples were discarded.

---

## Dataset

There is no single public dataset covering this problem: experimental heat-transfer data is scattered across journal supplements and is rarely available as a clean, ready-to-load table. To build and test the full pipeline, a **synthetic dataset** was generated, and it is clearly labelled as such everywhere (`data_source = synthetic_...` in every row).

| Property | Value |
|---|---|
| Rows | 20,000 |
| Fluids | 17 (liquids, gases, glycol/oil) |
| Flow configurations | 7 |
| Prandtl range | 0.66 – 931 |
| Target | `Nu_measured_synthetic` (physics value with multiplicative noise) |
| Noise | roughly 4–12% depending on regime; the level is an **assumption**, stored in `assumed_noise_sigma` |

**Columns include:** geometry, boundary condition, fluid and fluid class, pressure, `T_ref`, `T_wall`, property temperature, `ΔT`, characteristic length, `L/D`, velocity, ρ, μ, k, c_p, Pr, Pr_wall, viscosity ratio, Re, Ra, β, regime, `Nu_physics`, `Nu_measured_synthetic`, `h`, and a suggested split.

**Splits.** Besides train, validation and test, a dedicated **`test_unseen_fluid`** split holds out three fluids entirely (toluene, argon, ammonia) to test generalisation to substances the model has never seen. The geometry column also allows a **leave-one-geometry-out** evaluation.

---

## Hybrid ML model

**Idea: residual learning.** Instead of predicting `Nu` from scratch, the model predicts how much the physics result should be corrected:

```
target   = log10(Nu_measured) − log10(Nu_physics)
Nu_final = Nu_physics · 10^(model(x))
```

This keeps the output anchored to physics, which makes it far more robust outside the training data than a pure ML model.

**Features (16, all dimensionless or categorical).** `log Re`, `log Ra`, `log Pr`, `log L/D`, `ΔT/T`, the wall/bulk viscosity ratio, a one-hot encoding of the geometry (7) and of the fluid class (3: gas, liquid, incompressible liquid). The **fluid name is deliberately not a feature**: with only dimensionless groups as inputs, the model learns physics rather than memorising substances, which is why it transfers to unseen fluids. Quantities that do not apply to a geometry (for example `Ra` in a forced-flow case) are encoded with a sentinel value.

**Model.** XGBoost regressor (300 trees, learning rate 0.03, depth 4), saved together with its feature list, interval quantiles and domain bounds in a single `hybrid_bundle.joblib`, so the app has one artifact to load.

---

## Uncertainty and applicability domain

**Prediction interval (split conformal).** On the held-out validation set, the absolute log-error of the hybrid prediction is computed separately for each geometry family, and its 90% quantile `q` is stored. At prediction time:

```
interval = [ Nu_final · 10^(−q) ,  Nu_final · 10^(+q) ]
```

The interval is asymmetric in percent terms (the upper side is `10^q − 1`). The upper half-widths per geometry family in the shipped model:

| Geometry family | Upper half-width |
|---|---|
| Pipe | ±8.8% |
| Cylinder (cross-flow) | ±10.4% |
| Sphere | ±10.3% |
| Flat plate | ±11.4% |
| Vertical plate (natural) | ±14.0% |
| Horizontal cylinder (natural) | ±14.2% |

**Applicability domain.** The ML correction is applied only if all of the following hold; otherwise the app returns the physics result and states which condition failed:

1. The geometry was part of the training data.
2. The fluid was part of the training data.
3. Every numeric feature lies within the minimum–maximum range seen in training for that geometry.

---

## Experiments and results

All numbers below are on **synthetic data** and measure how well the system reproduces correlation-based targets with noise, not agreement with physical experiments. MAPE is the mean absolute percentage error of `Nu`.

### 1. Baselines versus hybrid (synthetic noise ≈ 5% is the floor)

Because the target is the physics value plus noise, the physics baseline sets a **noise floor** of about 5.2–5.4% MAPE that no model can meaningfully beat.

| Model | Test | Unseen fluids | Validation |
|---|---|---|---|
| Physics (noise floor) | 5.40 | 5.22 | 5.19 |
| Random forest (pure ML) | 6.90 | 8.13 | 6.65 |
| XGBoost (pure ML) | 6.86 | 7.47 | 6.61 |
| MLP (pure ML) | 6.29 | 6.34 | 6.22 |
| Hybrid, XGBoost residual | **5.43** | **5.24** | **5.22** |
| Hybrid, MLP residual | 5.73 | 5.60 | 5.48 |

The hybrid model stays at the noise floor and, unlike the pure-ML models, does not degrade on unseen fluids. Here the physics baseline *is* the ground truth by construction, so the learned correction is close to zero; this table demonstrates that the pipeline is sound, not that hybrid beats physics in reality.

### 2. Flawed-physics test (a more realistic situation)

To test whether the hybrid mechanism can actually repair a baseline, the physics was deliberately replaced by crude formulas (Dittus-Boelter-style pipe correlation, a simple `0.59·Ra^(1/4)` natural-convection law, a laminar plate formula), which creates a systematic gap to the target.

| Model | Test | Unseen fluids |
|---|---|---|
| Crude physics alone | 24.6 | 22.8 |
| Pure XGBoost | 7.0 | 8.2 |
| Hybrid on crude physics | **5.9** | **6.0** |

With an imperfect baseline, the hybrid cuts the error from about 25% to about 6% and beats pure ML, including on unseen fluids.

### 3. Leave-one-geometry-out (honest failure analysis)

Training on all geometries except one, with the correction learned on top of the crude baseline:

| Held-out geometry | Crude physics | Hybrid | Pure XGBoost |
|---|---|---|---|
| Vertical plate | 29.1 | **9.1** | 10.3 |
| Horizontal cylinder | 26.9 | 13.6 | 13.2 |
| Pipe | 29.0 | 27.3 | 152.0 * |
| Cylinder (cross-flow) | 12.2 | 16.9 | 18.6 |
| Flat plate | 19.3 | 27.4 | 32.2 |
| Sphere | 14.0 | 15.9 | 16.3 |

\* The pipe result is partly an artifact: `L/D` exists only for pipes, so holding the pipe out leaves the model with an input it has never seen.

**Takeaway:** a learned correction transfers to a new geometry only when that geometry resembles a trained one (the two natural-convection cases share the same crude law). For the other geometries it can make things worse. This is exactly why the applicability-domain guard exists: a geometry that is not in the training set receives the physics result only.

### 4. Interval calibration

| Set | Target coverage | Observed coverage |
|---|---|---|
| Test | 90% | 89.6% |
| Unseen fluids | 90% | 89.5% |

Coverage was measured in the calibration experiment (hybrid model on the crude baseline) and matches the nominal level, including for fluids absent from training. The shipped model uses the same calibration procedure on the physics baseline, but its coverage was not reported separately. On synthetic data the noise comes from one distribution per geometry, so real data will require re-calibration.

---

## Verification and worked examples

**Automated tests** (`test_fluidcalc.py`, 14 tests, all passing) check:

- the fully developed laminar limits (Nu = 3.66 and 4.364, reproduced as 3.69 and 4.40 for long pipes)
- continuity of `Nu` across the laminar/transitional/turbulent boundaries (`Re = 2300`, `4000`)
- that all 7 geometries run
- consistency with the dataset: on 60 random rows, `solve()` matches the dataset's `Nu_physics` to better than 0.1%
- error handling and out-of-range warnings for invalid inputs

**Independent hand calculations.** The values below were recomputed outside the application, directly from CoolProp properties and the published formulas. They are useful as reference points when you try the app.

| Case | Inputs | Key intermediate values | Physics result |
|---|---|---|---|
| **A. Water, pipe (const. wall T)** | 57 → 72 °C, D = 0.02 m, v = 1.5 m/s, L/D = 80, 2 bar | ρ = 984.8 kg/m³, μ = 4.892·10⁻⁴ Pa·s, Re = 6.04·10⁴, Pr = 3.158, f = 0.02008 | Nu = 293.8, h = 9519 W/(m²·K); outlet ≈ 62.7 °C, Q ≈ 11.3 kW |
| **B. Ethanol, pipe (const. wall T)** | 40 → 60 °C, D = 0.015 m, v = 2 m/s, L/D = 60, 1 bar | ρ = 772.1 kg/m³, Re = 2.83·10⁴, Pr = 13.05, f = 0.02398 | Nu = 280.7, h = 3008 W/(m²·K); outlet ≈ 43.3 °C, Q ≈ 2.33 kW |
| **C. Air, vertical plate (natural)** | 300 K → 350 K, L = 0.5 m, 1 bar | film T = 325 K, β = 3.083·10⁻³ 1/K, Pr = 0.7042, Ra = 3.93·10⁸ | Nu_L = 92.18, h = 5.202 W/(m²·K) |

In the deployed app, case B returned `Nu` (physics) = 280.7 and `Nu` (final) = 281.5, i.e. an ML correction of about +0.3%, consistent with the near-zero correction expected on synthetic data.

---

## Visualizations

**Sensitivity curves.** Every point is computed by the same physics + ML pipeline as the main result. The dashed line is the physics value, the solid line the final value, and the shaded band is the 90% interval (shown only where the ML was applied).

**3D response surface.** A 15 × 15 grid evaluated through the full pipeline over any two inputs, with the current input marked.

**Pipe 3D illustration.** For the constant-wall-temperature pipe, the bulk fluid temperature along the pipe is computed from an energy balance with the average `h`:

```
T_b(x) = T_w − (T_w − T_in) · exp( −h·π·D·x / (ṁ·c_p) )
Q      = ṁ · c_p · (T_out − T_in)
```

The outlet velocity profile is drawn as a parabola for laminar flow (`u/u_max = 1 − (r/R)²`) and as the 1/7 power law for turbulent flow (`u/u_max = (1 − r/R)^(1/7)`, with `U/u_max = 98/120`). The inlet temperature is taken as `T_ref`, and properties are evaluated at `T_ref`; the app warns when the fluid heats up so much that this becomes a poor approximation.

> **This is not a CFD simulation.** It is an analytical illustration based on the average heat-transfer coefficient. The pipe radius is exaggerated in the figure for visibility. Energy-balance results were cross-checked against `Q = h·A·ΔT_lm` and agree exactly.

---

## Repository structure

```
.
├── app.py                  # Streamlit interface (tabs, inputs, results)
├── fluidcalc.py            # Physics module: properties, correlations, step-by-step solution
├── hybrid.py               # ML correction, conformal interval, applicability-domain check
├── plots.py                # Plotly figures: sensitivity, 3D surface, pipe illustration
├── hybrid_bundle.joblib    # Trained model + feature list + interval quantiles + domain bounds
├── test_fluidcalc.py       # Unit tests for the physics module
└── requirements.txt        # Pinned dependencies (must match the training environment)
```

---

## Running locally and deployment

**Local**

```bash
git clone https://github.com/<your-username>/fluid-hybrid-calculator.git
cd fluid-hybrid-calculator
pip install -r requirements.txt
streamlit run app.py
```

**Tests**

```bash
pip install pytest
pytest test_fluidcalc.py
```

**Dependencies.** `requirements.txt` pins `xgboost`, `scikit-learn`, `joblib` and `CoolProp` to the versions used when the model was trained. This matters: a model serialised with one `xgboost` version may fail to load or behave differently under another. If you retrain the model, regenerate the bundle and update the pins.

**Deployment.** The app is deployed on Streamlit Community Cloud directly from this repository: connect the repository, choose `main` and `app.py`, and the platform installs `requirements.txt` automatically. The model bundle must be in the repository root next to `app.py`.

**Retraining.** The pipeline (dataset generation → splits → baselines → hybrid model → conformal calibration → bundle export) was developed in Google Colab. To retrain, regenerate or replace the dataset, refit the residual model on the same 16 features, recompute the per-geometry quantiles on a validation set, and save the bundle with the same keys.

---

## Limitations

- **Synthetic training data.** The target is generated from correlations, so reported errors describe how well the system reproduces those correlations plus assumed noise, **not** agreement with physical experiments. On this data the ML correction is close to zero, which is expected. The real value of the hybrid layer can only be measured on real measurements.
- **Assumed noise.** Noise levels are assumptions, not measured. Interval widths will change when the model is calibrated on real scatter.
- **New geometries.** As the leave-one-geometry-out analysis shows, a learned correction can worsen results on a geometry it has not seen. The applicability-domain guard prevents ML use in that case, but it is only as good as its feature ranges.
- **Average Nusselt number only.** The correlations give area-averaged `Nu`, not local distributions along a surface.
- **The pipe 3D tab is an illustration.** It uses an energy balance with a constant average `h`, not a CFD solution.
- **Scope.** Not covered: liquid metals (`Pr < 0.1`), boiling and condensation, finned or rough surfaces, mixed convection, and natural convection inside pipes.
- **Not a certified design tool.** Results are suitable for learning, estimation and prototyping. Safety-critical design needs validated, application-specific methods.

---

## Roadmap

- [ ] Validate against published experimental Nu–Re data and recalibrate the interval on real scatter
- [ ] Re-train the correction layer on real data and report real-world error
- [ ] Separate backend and frontend: a FastAPI `/calculate` endpoint consumed by the UI
- [ ] More problem types: pressure drop, LMTD/NTU heat exchangers, boundary layers
- [ ] Physics-informed neural network (PINN) variant
- [ ] PDF export of the step-by-step solution
- [ ] SI/Imperial unit switch

---

## References

- Bell, I. H., Wronski, J., Quoilin, S., Lemort, V. (2014). Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and the Open-Source Thermophysical Property Library CoolProp. *Industrial & Engineering Chemistry Research*, 53(6), 2498–2508.
- Gnielinski, V. (1976). New equations for heat and mass transfer in turbulent pipe and channel flow. *International Chemical Engineering*, 16, 359–368.
- Petukhov, B. S. (1970). Heat transfer and friction in turbulent pipe flow with variable physical properties. *Advances in Heat Transfer*, 6, 503–564.
- Churchill, S. W., Chu, H. H. S. (1975). Correlating equations for laminar and turbulent free convection from a vertical plate. *International Journal of Heat and Mass Transfer*, 18, 1323–1329.
- Churchill, S. W., Bernstein, M. (1977). A correlating equation for forced convection from gases and liquids to a circular cylinder in crossflow. *Journal of Heat Transfer*, 99, 300–306.
- Whitaker, S. (1972). Forced convection heat transfer correlations for flow in pipes, past flat plates, single cylinders, single spheres, and for flow in packed beds and tube bundles. *AIChE Journal*, 18, 361–371.
- Chen, T., Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD*, 785–794.
- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. (conformal prediction)

---

## Development notes

Built with Python, CoolProp, scikit-learn, XGBoost, Plotly and Streamlit. Development was AI-assisted (Claude by Anthropic) for code drafting and review; the project design decisions, data pipeline execution, model evaluation, verification of results and deployment were carried out by the author.

---
