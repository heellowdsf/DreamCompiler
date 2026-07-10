# Opt-in data augmentation

`augment(X, Y, when_below, up_to)` grows a small dataset with synthetic
samples -- but only when you ask, and only when it is safe.

## How it stays real (not garbage)

Synthetic samples are **SMOTE-style interpolations**: each new point is a
convex combination `a + t*(b - a)` of two real, same-class samples, plus a
tiny jitter (<=5% of each column's standard deviation). Because every point
lies between two real ones, it stays inside the real data's distribution --
the method interpolates, never extrapolates, so it cannot invent absurd
out-of-range values. Labels are carried through unchanged (same class), so
the input/output relationship is preserved.

## Guards

- Triggers only when real rows `< when_below`; otherwise the data is passed
  through untouched.
- Needs at least 2 real rows (and 2 per class to pair within a class); a
  single row can never seed augmentation.
- Compile-time warnings: `when_below < 1` (nothing to interpolate from),
  `up_to <= when_below` (would add nothing), `up_to > 100*when_below`
  (dataset would be dominated by synthetic points).
- Deterministic: fixed seed, so `augment` is reproducible.

## Run

    dream run augment_demo.dream

Both real-only and augmented runs reach full accuracy on this separable
task -- the point is that augmentation does not damage training. Fidelity
(synthetic values staying inside the real min/max per column, with matching
mean and std) is verified numerically during development.

## API

    augment(X, Y, when_below, up_to)     -> augmented X (real rows first)
    augment_y(X, Y, when_below, up_to)   -> augmented Y, aligned to X
    augment_count(XA)                    -> number of synthetic rows added
