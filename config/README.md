# Parameters and settings

## Files in this folder

| File | Content |
|---|---|
| `population_idm.json` | r-Safety spacing (`rsafety`) and the population IDM of each run of the recorded platoon (`pop_idm[run]`, each fitted on the other four runs); the population IDM is what a CAV forecasts the human drivers with |
| `driver_noise.json` | the two random terms of the human driver (spacing drift and acceleration noise, Ornstein–Uhlenbeck) |
| `mpc_weights.json` | weights of the MPC base controller |

Parameters fitted to individual drivers, the identification paths and the forecast-error tables are derived directly
from the recorded platoon and are not distributed; the code that needs them (the replay of the recorded experiment) is
marked in the source.

## Settings of the experiments

The settings of the paper's experiments, as used in the paper. Each row names where the value is set in the code.
Symbols as in the paper.

### Data and changes

| Item | Setting | Where |
|---|---|---|
| Data | five runs of a 25-car platoon (BJTU, 26 June 2021), 22 cars with position and speed at 10 Hz | `w2s/lib.py` |
| Sampling, vehicle length | 0.1 s; effective vehicle length 4.5 m | `w2s/lib.py` |
| Changes | 26 changes of the head between plateaus of 47, 37, 28, 19 and 9 km/h, 13 decelerations and 13 accelerations | `w2s/eval_identify.py` |

### Human drivers

| Item | Setting | Where |
|---|---|---|
| Car-following model | IDM per vehicle, exponent 4, desired speed fixed at 20 m/s; T, s0, a, b fitted on runs 1, 2, 4, 5 in 60-s pieces, run 3 for checking | calibration on the recorded data (not distributed) |
| Population model | one IDM for all drivers, for each run fitted on the other four runs; it is what a CAV forecasts with and it gives the deterministic part | `population_idm.json` |
| Spacing drift | Ornstein–Uhlenbeck offset of the desired spacing, σ = 4.5 m, τ = 30 s | `driver_noise.json` |
| Acceleration noise | Ornstein–Uhlenbeck, σ = 0.3 m/s², τ = 2 s | `driver_noise.json` |
| Fit of the two terms | decelerations and accelerations together, runs 1, 2, 4, 5: energy of the random part by group of cars (periods 10–25 s) and scatter of the pulse centres over 1, 2, 4, 8 drivers; least sum of squared log ratios over a grid | calibration on the recorded data (not distributed) |
| Draws | seed 0: drivers that follow the population model (the deterministic part); seeds 1–10 (Sections 5.3, 6.1) or 1–5 (Sections 6.2, 6.3): random drivers. A driver's random stream depends on (seed, run, change, rank) only, so every scenario meets the same draws | `platoon._rng` |

### CAVs and base controllers

| Item | Setting | Where |
|---|---|---|
| Limits | acceleration −3.5 to 2.0 m/s², jerk 5 m/s³, smallest gap 2 m | `w2s/lib.py` |
| r-Safety spacing | S_r(v) = d0 + τ_h v + r v²/(2 a_ref); r = 0.307 and a_ref = 3.41 m/s² from Zhou and Zhang (2026); d0 = 4.57 m and τ_h = 0.10 s fitted on quasi-steady following | `population_idm.json` |
| Spacing band | lag limit R_b(max(v1, v2)), borrowing limit R_b(v) | `transition.Cfg.band_up` |
| ACC | k_g = 0.5 s⁻², k_Δ = 1.8 s⁻¹ on the r-Safety spacing | `transition.Cfg` |
| CACC | Ploeg et al. (2011) on the r-Safety spacing: h u̇ + u = k_p e + k_d ė + u_{m−1}(t − θ), k_p = 0.2, k_d = 0.7, time gap h = dS_r/dv, V2V delay θ = 0.1 s; behind an HDV (no V2V) it falls back to the ACC | `platoon.py` (`PLOEG_*`) |
| MPC | linear MPC on the spacing deviation, weights q_g = 1, q_v = 4, r_a = 0.25, r_j = 0.25; step 0.5 s; horizon max(P, 5 s) with a preview P and 60 s with the whole future; the predecessor holds its last forecast speed beyond the preview; OSQP | `mpc.py`, `mpc_weights.json` |
| Relay | an announced plan reaches the CAV behind one step (0.1 s) later; no further latency is added | `platoon.HOP`, `announce_provider(lat=1)` |

### Planning (W²S)

| Item | Setting | Where |
|---|---|---|
| End state | the first level of the forecast held 2 s (1-s smoothed \|a\| ≤ 0.3 m/s²) and kept within 0.5 m/s for 8 s, at least 2 m/s from the CAV's own speed; behind a CAV, the end speed of its announced plan | `single_cav.first_plateau`, `platoon.PLAN_TGT` |
| Planning gate | plans only for changes of at least 2 m/s, and only before the predecessor has covered 10 % of its change | `transition.Cfg.gate`, `platoon.GO_FRAC` |
| Commitment check | commits only if the forecast made 2 s earlier held within 0.5 m/s | `FC_CHECK`, `FC_TOL` |
| Re-deciding | a CAV not yet committed decides again every 2 s while within 0.5 m/s of its initial speed | `RETRY_S`, `RETRY_DEP` |
| Re-planning | a new end state that differs by more than 2 m/s | `single_cav.RETGT` |
| Candidates, cost | AKS shapes inside the band; jerk weight c_j = 0.1 s²; horizon 60 s | `transition.Cfg`, `single_cav.H_S` |
| Joint schedule | exact recursion over the rank; 4 CAVs scheduled together (the deciding CAV and those behind it); 4 candidates kept per CAV behind and 8 for the deciding CAV, each plus feedback | `schedule.py` (`OPEN`, `KD`), `platoon.WS_NCAND` |
| Forecast behind HDVs | the trajectory of the head, or of the nearest CAV ahead with its announced plan, passed through the HDVs in between with the population IDM fitted on the other four runs, started from the predecessor's observed speed (2-s blend) | `platoon.py` (`FORECAST=idm`) |
| Forecast from the end state alone | departure front of 1.05 s per vehicle; typical transition 9.5 s | `platoon.ETA_DEP`, `platoon.T_PRIOR` |
| Buffer time | b = z s − μ, with μ and s the median and the robust standard deviation of the recorded timing error through n drivers on the other four runs; beyond the recorded chains (n > 8), s = σ_τ √n and μ proportional to n, both fitted on n ≤ 8; z = 1.28 (α = 10 %) unless stated | `platoon.protection`, `platoon.err_stats_loo` (`PROTECT_LOO=1`) |

### Execution

| Item | Setting | Where |
|---|---|---|
| Tracking | tracking gain k_v = 1.8 s⁻¹ | `transition.Cfg.kv` |
| Departure from the plan | ψ measured with the scales σ_g = 1 m and σ_v = 0.5 m/s; a departure ψ ≥ 3 reschedules the CAV | `transition.Cfg`, `D2` |
| Band guard | the command fades towards the base controller over the last 1 m before the lag limit and over the last half (at most 1 m) of the borrowing buffer | `single_cav.band_weight` |
| Hand-over | from plan to base controller over 2 s, jerk ≤ 1 m/s³; a plan is entered with jerk ≤ 1 m/s³ | `single_cav.HAND_S`, `HAND_J` |
| Buffer release | once the predecessor has covered 10 % of its change, its timing is observed: the forecast is moved to the observed timing and the rest of the transition is replanned, so the spacing the buffer added is spent | `platoon.py` |
| Predictive guard (Sections 6.2, 6.3) | the predecessor is predicted 4 s ahead as the forecast plus its present shortfall and the trend of the shortfall; the band guard's lower edge is applied to the predicted spacing deviation, and the command blends towards the smaller of feedback and plan | `EXIT_RULE=1 EXIT_MODE=pguard PG_MIN=1`, `platoon._pweight` |
| Tracking saturation (Sections 6.2, 6.3) | the tracking command is limited to −3.5 to 2.0 m/s², as the base controller | `TRK_SAT=1` |
| Buffer kept after release (Sections 6.2, 6.3) | replans after the release keep the spacing the buffer added | `REL_KEEP=1` |

The last three rows are options read from the environment; the experiments of Sections 6.2 and 6.3 switch them on.

### Transition window and measures

| Item | Setting | Where |
|---|---|---|
| Window opens | 5 s before the head, or a CAV that starts earlier, begins to change speed | evaluation of the experiments (not distributed) |
| Window closes | 5 s after the last vehicle has held the new speed within 0.5 m/s and its acceleration within 0.3 m/s² for 10 s, under the slowest scenario | `metrics.event_end` |
| Never settles | 30 s after the last vehicle has come within 1.5 km/h of the new speed (changes to 9 km/h) | `metrics.event_end` |
| Overlaps | changes that begin before the previous one has settled, or run into the next one, are kept and flagged (`how`) | evaluation of the experiments (not distributed) |
| Oscillation | Fourier transform of the acceleration over the window, no taper, padded with zeros to 1638.4 s; periods 10–25 s (Section 6.1 also 25–50 and 4–10 s); a_rms = √(ΣE / ΣT) | `metrics.band_energy_ev` |
| Transition | the central part of a change: from the last time the vehicle has covered less than 10 % of it to the first time it has covered 90 % (1-s moving average of the speed), so that its speed changes by 0.8 Δv | `metrics.extent_back` |
| Duration, distance | time and distance of the transition, on the drivers that follow the population model | `metrics.extent_back` |
| Energy, shape factor | E = ∫ a² dt over the transition and κ = Δt E / (0.8 Δv)² (κ ≥ 1, = 1 for a constant acceleration); energy ratios are geometric means, for which E ratio = κ ratio × duration ratio exactly. Table 3: E_m/E_{m−1}, the CAV against its predecessor (whole future); Table 4: E/E_HDV, the HDVs behind the CAVs (ranks ≥ 5) against the all-HDV platoon, on the drivers that follow the population model (seed 0) | evaluation of the experiments (not distributed) |
| Fuel economy | fuel rate of a mid-size passenger car on a flat road (`midBase`), summed over vehicles and episodes | `metrics.fuel` |
| Travel time | time to cover the stretch the same vehicle covers in the window in the all-HDV platoon; beyond the end of a run the vehicle continues at the new speed; the two changes cut by the end of the record are left out | evaluation of the experiments (not distributed) |
| Time to collision | counted only while the vehicle is faster than 2 m/s and closing in on its predecessor; Table 3 gives the smallest value (median over changes) | `metrics.ttc` |
| Conflicts | share of vehicle-episodes whose smallest time to collision falls below 1.5 s | evaluation of the experiments (not distributed) |
