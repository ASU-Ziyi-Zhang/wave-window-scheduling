# Wave-window scheduling (W²S)

Code and parameters of the method in

> *Wave-window scheduling of speed transitions: a planning layer for connected and automated vehicles in stochastic
> mixed traffic* (submitted).

A connected and automated vehicle (CAV) that learns early where its predecessor's change of speed will end can spread
its own change over a longer time and pass a weaker wave to the vehicles behind. The **wave window** is the time
between when the vehicle learns where its predecessor's change ends and when its spacing no longer lets it wait.
**Wave-window scheduling** is a planning layer above the CAV's base controller (ACC, CACC or MPC): inside this window it
decides when each change of speed starts, how long it lasts and what shape it takes, announces the plan to the CAVs
behind, and holds part of the window back as a buffer against the timing error of the information.

## Contents

```
w2s/        the method
  lib.py            constants, the r-Safety spacing, the intelligent driver model (IDM)
  transition.py     settings and the candidate transitions (adaptive kinematic smoothing) inside the spacing band
  rollout.py        rollouts of a CAV under feedback or tracking a planned transition, safety filter
  single_cav.py     one CAV: forecast, target, plan, commitment, tracking, hand-back to the base controller
  schedule.py       joint scheduling of a string of CAVs
  platoon.py        a platoon in closed loop: base controllers, forecasts through human drivers, announced plans,
                    buffer time
  mpc.py            the linear MPC base controller
  random_driver.py  the stochastic human driver (IDM with a drifting desired spacing and an acceleration noise)
  metrics.py        duration and energy of a transition, wave-band acceleration, fuel economy, time to collision
  identify.py, eval_identify.py   when a follower can identify where a change ends
config/     calibrated parameters and the settings of the paper (config/README.md)
examples/   a self-contained example that needs no recorded data
```

## Quick start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python examples/single_cav_example.py          # add --plot for a figure
```

The example lets one CAV follow a synthetic predecessor that slows from 47 to 19 km/h, once under ACC alone and once
with W²S layered on it, and prints the duration of the CAV's transition and the share of the predecessor's
acceleration energy that it passes on:

```
predecessor: 47 -> 19 km/h, transition 4.9 s
ACC alone : transition   5.3 s, energy passed on / received 0.91, smallest gap   6.4 m, plans committed 0
ACC + W2S : transition  11.1 s, energy passed on / received 0.45, smallest gap   6.3 m, plans committed 1
```

## Data

The experiments of the paper replay a controlled 25-vehicle platoon experiment recorded at Beijing Jiaotong
University. The recorded data were provided by their original authors and are not distributed here; neither are the
parameters fitted to individual drivers and the tables derived directly from the recordings. The parts of the code
that replay the recorded experiment need these files and say so where they load them.

## License

MIT (see `LICENSE`).
