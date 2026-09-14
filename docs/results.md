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
| 6 | Does the pipeline run inside RAMMS? | yes; MuJoCo-trained models keep their ranking there |

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

## 6. RAMMS backend, first collection

8 rovers, 600 s, seed 0 settings (clutter only), collected inside RAMMS through
URLab with `--backend ramms` (3.2 min, 3x real time). The models trained in
MuJoCo for experiment 3 (seed 0) were then scored on this RAMMS data.

| Model | On the RAMMS run | On its own MuJoCo test split |
|---|---:|---:|
| Local only | 0.810 | 0.877 |
| FedAvg | 0.867 | 0.924 |
| Centralized | 0.896 | 0.942 |

- The ranking carries over to RAMMS. Scores drop by about 0.05, but the MuJoCo
  test split is the tail of the same run the models trained on, while the RAMMS
  run is entirely new, so part of the drop is not the simulator. Scoring on a
  fresh MuJoCo run would separate the two.
- Rangefinder readings are identical to standalone MuJoCo for the same pose, and
  trajectories under identical commands differ by about 1 cm after 2 s.

## Limits

- One narrow task that rangefinders make fairly easy, and one exploration
  policy shared by every rover.
- Few seeds: five per condition, three for 16 rovers.
- Client processes are separated by what each is told, not by operating-system
  permissions, and evaluation reads every rover's test split.
- The label definition moves absolute scores and the size of the gaps; compare
  methods only within one label type.

## Next

- Train and federate on RAMMS data, then add camera input and crowds.
- Look rover by rover at where the shared model hurts before trying heavier
  personalization (fewer fine-tuning epochs, shared body with per-rover heads,
  clustering rovers by speed).
- Partial participation and dropped rounds, closer to real robots on flaky
  Wi-Fi.
