# Results

Every experiment so far, in the order it was run. Each one answers a question
the previous one raised.

All numbers are test AUPRC (area under the precision-recall curve) on the final
20% of each rover's run, averaged over rovers. Multi-seed experiments collect a
fresh dataset per seed; ± values are standard deviations across seeds, and
"paired" differences compare two methods within the same seed. Unless noted,
fleets have 8 rovers, 600 simulated seconds each, clutter from 0.1 to 1.2
obstacles per square metre, and the time label (collision within 0.5 s).
Compute was one 8-thread CPU.

| # | Question | Result |
|---|---|---|
| 1 | Does federation help at all? | yes: 0.880 against 0.850 local-only |
| 2 | Does it hold over seeds, and does FedProx help? | yes, in 5 of 5 seeds; FedProx no |
| 3 | Can communication rounds be cut? | 5x fewer rounds, same accuracy; FedProx no at any strength |
| 4 | What happens when rovers really differ? | federation still wins everywhere; slow rovers gain most; fine-tuning hurts |
| 5 | How much of the speed effect was the label? | most of it |
| 6 | Does it hold inside RAMMS? | yes, and models move between simulators with almost no loss |
| 7 | Does the front camera help? | yes for local-only and FedAvg (+0.03 to +0.06), not centralized |
| 8 | What do pedestrians change? | every score drops; federation matters more in MuJoCo; in RAMMS the camera stops helping |
| 9 | Does the camera model need to see motion? | no: a window of frames matches a single frame everywhere |
| 10 | Does the prediction make the fleet safer? | yes: collisions roughly halve, and the shared model beats local-only per metre |

## 1. First run

Single seed, 20 rounds of 1 local epoch, baselines 20 epochs (10 min).

```bash
ramms-fleet-collect --rovers 8 --seconds 600 --out data/run0
ramms-fleet-baseline --data data/run0 --out results/run0 --epochs 20
ramms-fleet-federate --data data/run0 --out results/run0 --rounds 20
ramms-fleet-eval --data data/run0 --results results/run0
```

| Method | AUPRC |
|---|---:|
| Local only | 0.850 |
| FedAvg | 0.880 |
| Centralized | 0.941 |

FedAvg beat local-only on 6 of 8 rovers, most on rovers whose own model was
weak, and its federated evaluation AUPRC was still rising (0.607 after round 1,
0.821 at round 10, 0.884 at round 20).

## 2. Five seeds, 50 rounds, FedProx

1 local epoch per round, baselines 50 epochs (163 min).

```bash
ramms-fleet-sweep --seeds 0 1 2 3 4 --rounds 50 --proximal-mus 0.1 --evaluate-every 5 \
    --data-root data/sweep --out results/sweep
```

| Method | AUPRC |
|---|---:|
| Local only | 0.821 ± 0.044 |
| FedProx (μ = 0.1) | 0.882 ± 0.026 |
| FedAvg | 0.890 ± 0.023 |
| Centralized | 0.907 ± 0.023 |

| Paired comparison | Difference | Seeds in favour |
|---|---:|---:|
| FedAvg - local only | +0.068 ± 0.022 | 5 of 5 |
| Centralized - FedAvg | +0.017 ± 0.003 | 5 of 5 |
| FedProx - FedAvg | -0.008 ± 0.004 | 0 of 5 |

- Federated training beats every rover training alone, in every seed, and
  closes about 80% of the gap to pooling the data.
- FedProx is slightly but consistently worse. With one local epoch per round,
  clients barely drift from the global model, so the proximal term mostly
  slows learning.
- Federated evaluation AUPRC at rounds 10, 25, and 50: FedAvg 0.774, 0.855,
  0.889; FedProx 0.742, 0.841, 0.880.

Mean over seeds by rover (rover index follows clutter):

| Rover | Clutter | Local only | FedAvg | FedProx | Centralized |
|------:|--------:|-----------:|-------:|--------:|------------:|
| 0 | 0.10 | 0.878 | 0.917 | 0.915 | 0.924 |
| 1 | 0.26 | 0.847 | 0.896 | 0.894 | 0.907 |
| 2 | 0.41 | 0.760 | 0.856 | 0.847 | 0.883 |
| 3 | 0.57 | 0.817 | 0.878 | 0.865 | 0.903 |
| 4 | 0.73 | 0.803 | 0.886 | 0.873 | 0.877 |
| 5 | 0.89 | 0.813 | 0.888 | 0.881 | 0.928 |
| 6 | 1.04 | 0.842 | 0.886 | 0.874 | 0.901 |
| 7 | 1.20 | 0.811 | 0.911 | 0.904 | 0.928 |

## 3. Five local epochs, 10 rounds

Same datasets and total training as experiment 2 (50 epochs per rover), a fifth
of the communication rounds (87 min). The baselines reproduce exactly.

```bash
ramms-fleet-sweep --seeds 0 1 2 3 4 --rounds 10 --local-epochs 5 --proximal-mus 0.01 0.1 1.0 \
    --evaluate-every 2 --data-root data/sweep --out results/sweep-e5
```

| Method | AUPRC | Paired vs FedAvg | Seeds beating FedAvg |
|---|---:|---:|---:|
| Local only | 0.821 ± 0.044 | -0.067 ± 0.020 | 0 of 5 |
| FedProx μ = 1.0 | 0.645 ± 0.048 | -0.244 ± 0.035 | 0 of 5 |
| FedProx μ = 0.1 | 0.848 ± 0.037 | -0.041 ± 0.013 | 0 of 5 |
| FedProx μ = 0.01 | 0.886 ± 0.027 | -0.003 ± 0.003 | 1 of 5 |
| FedAvg | 0.889 ± 0.026 | | |
| Centralized | 0.907 ± 0.023 | +0.018 ± 0.005 | 5 of 5 |

- FedAvg with 5 local epochs and 10 rounds matches 1 epoch and 50 rounds
  (-0.001 ± 0.003 per seed): this task tolerates 5x less communication.
- FedProx does not help at any strength. At μ = 1.0 it holds clients so close
  to the global model that 10 rounds are not enough (0.637 federated evaluation
  AUPRC at round 10).
- Client drift is not what limits FedAvg here. Clutter changes how often rovers
  collide, but an imminent collision looks the same to the sensors in every
  arena, which motivated experiment 4.

## 4. Rovers that differ

Rovers also get their own cruise speed (0.15 to 0.60 m/s) and sensor noise
(rangefinder σ up to 0.20 m, accelerometer 2 m/s², gyro 0.3 rad/s), shuffled
across rovers independently of clutter. "FedAvg + fine-tune" personalizes the
finished FedAvg model with 5 epochs on each rover's own data. 10 rounds x 5
local epochs, FedProx μ = 0.01 and 0.1 (about 4 h, `scripts/overnight_heterogeneity.sh`).

| Condition | Seeds | Local only | FedAvg | FedAvg + fine-tune | Centralized |
|---|---:|---:|---:|---:|---:|
| Clutter only | 5 | 0.821 ± 0.044 | 0.889 ± 0.026 | 0.874 ± 0.029 | 0.907 ± 0.023 |
| Clutter + speed | 5 | 0.827 ± 0.033 | 0.871 ± 0.026 | 0.862 ± 0.036 | 0.906 ± 0.028 |
| Clutter + noise | 5 | 0.769 ± 0.026 | 0.833 ± 0.022 | 0.818 ± 0.027 | 0.839 ± 0.023 |
| All three | 5 | 0.755 ± 0.030 | 0.816 ± 0.008 | 0.802 ± 0.014 | 0.825 ± 0.009 |
| All three, 16 rovers | 3 | 0.771 ± 0.004 | 0.827 ± 0.026 | 0.814 ± 0.022 | 0.844 ± 0.019 |

| Paired comparison | Clutter | Speed | Noise | All three | 16 rovers |
|---|---:|---:|---:|---:|---:|
| FedAvg - local only | +0.067 | +0.044 | +0.065 | +0.061 | +0.056 |
| Centralized - FedAvg | +0.018 | +0.034 | +0.005 | +0.008 | +0.017 |
| Fine-tune - FedAvg | -0.014 | -0.009 | -0.015 | -0.014 | -0.013 |
| FedProx μ 0.01 - FedAvg | -0.003 | +0.002 | -0.002 | -0.003 | -0.000 |
| FedProx μ 0.1 - FedAvg | -0.041 | -0.027 | -0.025 | -0.023 | -0.019 |

- FedAvg beats local-only in all 23 seeds.
- Speed differences open the widest gap to centralized training (0.034 against
  0.018 with clutter alone). Experiment 5 shows most of that came from the
  time-based label.
- Slow rovers gain the most. The slowest third collides 7.6 times per minute
  against 17.6 for the fastest third, so their own data has few positives:
  local-only 0.469, FedAvg 0.620, centralized 0.609 (all three differences).
- Fine-tuning lowers mean AUPRC in every condition. It helps fast rovers
  slightly but overfits the few positives of slow ones.
- FedProx never beats FedAvg beyond seed noise.

The script also renders videos to `results/overnight/videos/`: the fleet colored
by FedAvg's predicted collision risk, and local-only against FedAvg for the
slowest, fastest, and noisiest rovers.

## 5. Distance-based labels

The time label covers four times more ground for the fastest rover than for the
slowest. This relabels the experiment 4 datasets as a collision within 0.15 m of
travel (equal to 0.5 s at the default 0.3 m/s) and reruns local-only, FedAvg,
fine-tuning, and centralized training (5 seeds each, about 100 min,
`scripts/distance_labels.sh`).

| Condition | Label | Local only | FedAvg | FedAvg + fine-tune | Centralized | Centralized - FedAvg |
|---|---|---:|---:|---:|---:|---:|
| Clutter only | time | 0.821 | 0.889 | 0.874 | 0.907 | +0.018 ± 0.005 |
| Clutter only | distance | 0.873 | 0.923 | 0.915 | 0.955 | +0.031 ± 0.015 |
| Clutter + speed | time | 0.827 | 0.871 | 0.862 | 0.906 | +0.034 ± 0.015 |
| Clutter + speed | distance | 0.864 | 0.897 | 0.900 | 0.935 | +0.038 ± 0.009 |
| Clutter + noise | time | 0.769 | 0.833 | 0.818 | 0.839 | +0.005 ± 0.010 |
| Clutter + noise | distance | 0.857 | 0.903 | 0.892 | 0.913 | +0.011 ± 0.012 |
| All three | time | 0.755 | 0.816 | 0.802 | 0.825 | +0.008 ± 0.008 |
| All three | distance | 0.794 | 0.857 | 0.845 | 0.871 | +0.014 ± 0.006 |

- Most of the speed effect was the label. Adding speed differences widened the
  gap to centralized training by +0.016 with time labels but only +0.006 with
  distance labels, within about one standard deviation.
- FedAvg still beats local-only in all 20 seeds, and slow rovers still gain the
  most (all three differences: local-only 0.624, FedAvg 0.759).
- Distance labels raise every score, since rangefinder readings map more
  directly to distance than to time, so compare methods within one label type.
- Fine-tuning is roughly neutral in the speed condition (+0.003 ± 0.008) and
  still lowers AUPRC elsewhere.

## 6. The same fleet inside RAMMS

The experiment 3 settings (clutter only, 10 rounds x 5 local epochs, seeds 0-4,
same arenas), with every run collected inside the RAMMS editor through URLab
(`scripts/ramms_federated.sh`, 15 min of collection at 3.6x real time, then
training). Models were also scored across simulators on each seed's test split.

| Trained → scored | Local only | FedAvg | Centralized |
|---|---:|---:|---:|
| MuJoCo → MuJoCo | 0.821 ± 0.044 | 0.889 ± 0.026 | 0.907 ± 0.023 |
| RAMMS → RAMMS | 0.832 ± 0.033 | 0.887 ± 0.022 | 0.919 ± 0.014 |
| MuJoCo → RAMMS | | 0.886 ± 0.017 | 0.912 ± 0.019 |
| RAMMS → MuJoCo | | 0.880 ± 0.028 | 0.908 ± 0.015 |

- Federated training behaves the same inside RAMMS: FedAvg beats local-only
  by +0.054 ± 0.011 (5 of 5 seeds) and trails centralized by 0.033 ± 0.011.
- Models move between simulators with almost no loss. Rangefinders match
  standalone MuJoCo exactly for the same pose, and trajectories drift apart by
  about 1 cm over 2 s.
- An earlier single-seed check suggested a 0.05 drop on RAMMS data; that came
  from scoring an entirely new run against the tail of the training run, not
  from the simulator.

## 7. The front camera as an input

RAMMS rendered each rover's front camera in step with the simulation
(`render: "sync"`: every frame matches its step, collection at 0.42x real
time). Five 300 s seeds were collected at 64 x 48 grayscale; models see frames
average-pooled to 32 x 24. Seeds 0-2 were trained with features only, camera
only, and both, on identical samples (`scripts/ramms_camera.sh`, about 3 h).

| Inputs | Local only | FedAvg | FedAvg + fine-tune | Centralized |
|---|---:|---:|---:|---:|
| Features | 0.740 ± 0.051 | 0.846 ± 0.055 | 0.822 ± 0.051 | 0.901 ± 0.029 |
| Camera | 0.765 ± 0.069 | 0.839 ± 0.074 | 0.827 ± 0.051 | 0.812 ± 0.068 |
| Both | 0.802 ± 0.065 | 0.875 ± 0.053 | 0.848 ± 0.063 | 0.887 ± 0.028 |

| Adding the camera to the features (same seed) | Difference | Seeds improved |
|---|---:|---:|
| Local only | +0.062 ± 0.023 | 3 of 3 |
| FedAvg | +0.030 ± 0.005 | 3 of 3 |
| Centralized | -0.013 ± 0.013 | 0 of 3 |

- The camera alone is about as good as the rangefinders once federated.
- Adding it helps the models trained on less data (local-only and FedAvg) but
  not centralized training, so with the camera FedAvg comes within 0.012 of
  centralized.
- Three seeds, and camera-only models vary a lot between seeds; treat the
  sizes as provisional.
- Getting clean frames needed two editor fixes: lights in the generated level
  (it renders black otherwise) and turning off "Use Less CPU when in
  Background", which throttles rendering. Rendering also needs a CPU that is
  not saturated by training.

## 8. Pedestrians

Each arena gained 0 to 7 scripted pedestrians (`--pedestrians 0 7`), a
different count per rover, shuffled independently of clutter. Pedestrians are
capsules on velocity-driven slide joints that walk between random goals at
0.15 to 0.35 m/s, pause, and steer around walls, obstacles, and each other;
half also steer around the rover. They collide, block rangefinders, and render
on camera in both simulators, and for the same seed they walked the same paths
in MuJoCo and RAMMS (within a millimetre over a 5 s check).
`scripts/crowds.sh` ran both parts in about 2.5 h.

**MuJoCo, 5 seeds x 600 s, features only.** Arena layouts and seeds match
experiment 3, so the comparison is paired by seed.

| Arenas | Local only | FedAvg | FedAvg + fine-tune | Centralized |
|---|---:|---:|---:|---:|
| No pedestrians (experiment 3) | 0.821 ± 0.044 | 0.889 ± 0.026 | 0.874 ± 0.029 | 0.907 ± 0.023 |
| 0 to 7 pedestrians | 0.689 ± 0.023 | 0.787 ± 0.022 | 0.762 ± 0.021 | 0.816 ± 0.016 |

| Paired difference | No pedestrians | With pedestrians |
|---|---:|---:|
| FedAvg - local | +0.067 ± 0.020 | +0.097 ± 0.020 (5 of 5 seeds) |
| Centralized - FedAvg | +0.018 ± 0.005 | +0.030 ± 0.010 |
| Fine-tune - FedAvg | -0.014 ± 0.005 | -0.025 ± 0.008 |

| Pedestrians in the arena (all rovers, all seeds) | Collisions/min | With a pedestrian | Local only | FedAvg | Centralized |
|---|---:|---:|---:|---:|---:|
| 0 to 1 | 10.5 | 0.5 | 0.767 | 0.868 | 0.878 |
| 2 to 4 | 15.4 | 4.9 | 0.700 | 0.805 | 0.838 |
| 5 to 7 | 22.6 | 11.3 | 0.627 | 0.714 | 0.754 |

- Pedestrians were involved in 37% of collisions (within 0.25 m of the rover
  at contact).
- Every method drops, local-only most (-0.132 ± 0.058 against -0.102 for
  FedAvg), so FedAvg's lead over training alone grew by +0.030 ± 0.029 (4 of 5
  seeds).
- The gap to centralized widens with crowd size, from 0.010 to 0.039.
- The federated score was still rising at round 10 (0.779 at round 8, 0.789 at
  round 10); more rounds may narrow the gap.
- Collection slowed from about 15x to 6x real time.

**RAMMS, 3 seeds x 300 s with the front camera.** Trained as in experiment 7,
every input choice on the same samples. Collection ran at 0.58x real time.
Pedestrians were involved in 31% of collisions.

| Inputs | Local only | FedAvg | FedAvg + fine-tune | Centralized | FedAvg - local |
|---|---:|---:|---:|---:|---:|
| Features | 0.743 ± 0.020 | 0.799 ± 0.052 | 0.789 ± 0.040 | 0.835 ± 0.040 | +0.056 ± 0.040 (3 of 3) |
| Camera | 0.736 ± 0.046 | 0.772 ± 0.084 | 0.780 ± 0.064 | 0.756 ± 0.074 | +0.037 ± 0.038 (3 of 3) |
| Both | 0.744 ± 0.039 | 0.784 ± 0.084 | 0.788 ± 0.052 | 0.823 ± 0.059 | +0.040 ± 0.060 (2 of 3) |

| Adding the camera to the features (same seed) | No pedestrians (exp. 7) | With pedestrians | Seeds improved |
|---|---:|---:|---:|
| Local only | +0.062 ± 0.023 | +0.002 ± 0.018 | 1 of 3 |
| FedAvg | +0.030 ± 0.005 | -0.015 ± 0.032 | 1 of 3 |
| Centralized | -0.013 ± 0.013 | -0.013 ± 0.022 | 1 of 3 |

- With pedestrians the camera no longer helps. A possible reason: the features
  cover the last four steps, so they show a pedestrian closing in, while camera
  models see only the latest frame.
- Seed 1 with features and camera is the first seed in any experiment where
  FedAvg lost to local-only training (by 0.028).
- Three seeds with large spread; treat these sizes as provisional.
- The pedestrians are not RAMMS's own crowd. RammsCrowd agents are Unreal Mass
  entities with no interface for reading their positions from outside the
  editor, and they would exist only in RAMMS.

## 9. A window of camera frames

Experiment 8 left an explanation to test: a camera model sees one frame, so it
cannot see a pedestrian moving, while the features cover four steps.
`--inputs camera-history` and `both-history` stack the whole feature window (4
frames) as channels instead. Both RAMMS datasets were retrained, no new
collection (`scripts/frame_history.sh`, about 2.5 h).

Arenas with pedestrians (3 seeds):

| Inputs | Local only | FedAvg | FedAvg + fine-tune | Centralized |
|---|---:|---:|---:|---:|
| Features | 0.743 ± 0.020 | 0.799 ± 0.052 | 0.789 ± 0.040 | 0.835 ± 0.040 |
| Camera, 1 frame | 0.736 ± 0.046 | 0.772 ± 0.084 | 0.780 ± 0.064 | 0.756 ± 0.074 |
| Camera, 4 frames | 0.732 ± 0.046 | 0.781 ± 0.079 | 0.773 ± 0.056 | 0.707 ± 0.089 |
| Both, 1 frame | 0.744 ± 0.039 | 0.784 ± 0.084 | 0.788 ± 0.052 | 0.823 ± 0.059 |
| Both, 4 frames | 0.742 ± 0.005 | 0.794 ± 0.072 | 0.775 ± 0.032 | 0.789 ± 0.041 |

Arenas without pedestrians, the experiment 7 data (3 seeds):

| Inputs | Local only | FedAvg | FedAvg + fine-tune | Centralized |
|---|---:|---:|---:|---:|
| Features | 0.740 ± 0.051 | 0.846 ± 0.055 | 0.822 ± 0.051 | 0.901 ± 0.029 |
| Camera, 1 frame | 0.765 ± 0.069 | 0.839 ± 0.074 | 0.827 ± 0.051 | 0.812 ± 0.068 |
| Camera, 4 frames | 0.736 ± 0.084 | 0.833 ± 0.044 | 0.810 ± 0.036 | 0.819 ± 0.085 |
| Both, 1 frame | 0.802 ± 0.065 | 0.875 ± 0.053 | 0.848 ± 0.063 | 0.887 ± 0.028 |
| Both, 4 frames | 0.773 ± 0.085 | 0.875 ± 0.045 | 0.839 ± 0.053 | 0.842 ± 0.088 |

| Adding camera frames to the features, FedAvg (same seed) | 1 frame | 4 frames |
|---|---:|---:|
| Without pedestrians | +0.030 ± 0.005 (3 of 3) | +0.029 ± 0.012 (3 of 3) |
| With pedestrians | -0.015 ± 0.032 (1 of 3) | -0.005 ± 0.015 (1 of 3) |

- Frame history changes nothing that matters. Where the single frame helped it
  still helps by the same amount, and where it did not, history does not
  rescue it.
- So the explanation offered in experiment 8, that camera models cannot see
  movement, is wrong. What the camera adds with pedestrians is below what 3
  seeds with spreads of ±0.03 to ±0.08 can resolve.
- Frame history hurts centralized training in both datasets (-0.047 and
  -0.058), the case with the most data and the most parameters to fit; four
  channels quadruple the first convolution's inputs.
- Samples now require every frame of the window to be fresh, not only the
  latest, which drops 24 more samples per dataset out of about 100,000.
  Earlier single-frame runs used the looser rule.

## 10. Letting the model drive

Every experiment above scores a model that never touches the robot. Here each
rover runs its own model on board: at each control step it forms the command
the exploration policy wants, scores it, and above a risk threshold keeps 25%
of its speed and turns away from the nearer side. Recovery after a bump is left
alone, and arenas, sensors, noise, and the policy are unchanged, so the model is
the only difference. Standing still would be trivially safe, so runs report
distance covered as well (`scripts/guard.sh`, 5 seeds x 8 rovers x 600 s,
about 20 min).

Clutter-only arenas with the experiment 3 models, threshold 0.5:

| Guard | Collisions/min | Collisions/100 m | Metres/min | Steps guarded |
|---|---:|---:|---:|---:|
| None | 11.8 ± 1.0 | 95.0 ± 9.2 | 12.5 ± 0.2 | 0% |
| Local only | 6.3 ± 2.7 | 53.2 ± 25.5 | 12.1 ± 0.7 | 15% |
| FedAvg | 5.8 ± 2.2 | 49.2 ± 21.8 | 12.1 ± 0.7 | 15% |
| Centralized | 5.1 ± 2.8 | 42.7 ± 26.8 | 12.3 ± 0.9 | 17% |

Arenas with pedestrians and the experiment 8 models. The threshold matters more
than the model, so it was swept for FedAvg first:

| FedAvg threshold | Collisions/min | Collisions/100 m | Metres/min | Steps guarded |
|---|---:|---:|---:|---:|
| No guard | 16.9 ± 1.2 | 142.6 ± 10.9 | 11.8 ± 0.1 | 0% |
| 0.7 | 12.6 ± 2.9 | 107.5 ± 27.8 | 11.8 ± 0.4 | 14% |
| 0.5 | 11.4 ± 2.1 | 98.2 ± 20.8 | 11.7 ± 0.4 | 17% |
| 0.3 | 8.2 ± 1.7 | 72.4 ± 16.7 | 11.4 ± 0.5 | 23% |
| 0.15 | 7.9 ± 1.8 | 80.4 ± 15.4 | 9.8 ± 1.1 | 38% |

At threshold 0.3, the best of those:

| Guard | Collisions/min | Collisions/100 m | Metres/min | Steps guarded |
|---|---:|---:|---:|---:|
| None | 16.9 ± 1.2 | 142.6 ± 10.9 | 11.8 ± 0.1 | 0% |
| Local only | 9.2 ± 1.5 | 84.9 ± 16.9 | 11.0 ± 0.4 | 25% |
| FedAvg | 8.2 ± 1.7 | 72.4 ± 16.7 | 11.4 ± 0.5 | 23% |
| Centralized | 7.9 ± 1.7 | 78.3 ± 18.3 | 10.2 ± 0.6 | 35% |

- Prediction pays off in behaviour, not only in AUPRC: collisions per 100 m
  roughly halve, in 5 of 5 seeds in both arena sets, while distance per minute
  falls by 3 to 6%.
- The shared model beats local-only per metre in arenas with pedestrians at
  threshold 0.3: 72.4 against 84.9 per 100 m, better in 5 of 5 seeds. In
  clutter-only arenas at threshold 0.5 the same comparison is 49.2 against
  53.2, but only 2 of 5 seeds, so that one is noise.
- Centralized and FedAvg end up close, and which wins depends on the metric:
  centralized has fewer collisions per minute (7.9 against 8.2) because it
  guards 35% of steps against 23%, but is worse per metre travelled.
- A model that triggers more often looks safer per minute while covering less
  ground. Collisions per 100 m is the fair comparison, and comparing models at
  one threshold still mixes ranking quality with how calibrated the scores are.
- The most cautious setting (0.15) is the first that costs real progress: 17%
  less distance for no gain per metre.
- A control run with the guard disabled by threshold reproduces the unguarded
  numbers exactly, so the harness itself changes nothing.

## Limits

- One narrow task that rangefinders make fairly easy, and one exploration
  policy shared by every rover.
- Few seeds: five per condition, three for 16 rovers.
- Client processes are separated by what each is told, not by operating-system
  permissions, and evaluation reads every rover's test split.
- The label definition moves absolute scores and the size of the gaps; compare
  methods only within one label type.
- Pedestrians are scripted capsules with a simple steering rule.
- Three RAMMS seeds cannot resolve differences smaller than about 0.05 AUPRC,
  which is most of what the camera does or does not add.
- The guard is one fixed rule (brake and turn) at one threshold per run, and
  the models were not calibrated, so guard results compare policies, not
  predictors alone.

## Next

- RAMMS's own crowd: expose Mass agent positions from RammsCrowd so its
  animated pedestrians drive the physics bodies in place of the scripted
  walkers.
- More federated rounds with pedestrians, and more RAMMS seeds, before drawing
  any further conclusion about the camera.
- Look rover by rover at where the shared model hurts before trying heavier
  personalization (fewer fine-tuning epochs, shared body with per-rover heads,
  clustering rovers by speed).
- Partial participation and dropped rounds, closer to real robots on flaky
  Wi-Fi.
- Robustness to a broken rover (dead rangefinder, stuck bumper, flipped
  labels) and aggregation that resists it.
- Calibrate scores per rover, or tune the threshold per model, so guard runs
  compare predictors rather than caution levels.
