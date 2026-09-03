use super::*;
use crate::oracle::best_penalized;
use crate::{CostL2, SignalView};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

fn l2(signal: &[f64]) -> CostL2 {
    CostL2::fit(SignalView::new(signal, signal.len(), 1).unwrap()).unwrap()
}

#[test]
fn finds_known_penalized_partition() {
    let cost = l2(&[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, -5.0, -5.0, -5.0]);
    let result = Pelt::new(1, 1)
        .unwrap()
        .predict_penalty(&cost, 1.0)
        .unwrap();
    assert_eq!(result.breakpoints, [3, 6, 9]);
    assert_eq!(result.segment_cost, 0.0);
    assert_eq!(result.objective, 2.0);
}

#[test]
fn large_penalty_selects_no_changes() {
    let cost = l2(&[0.0, 0.0, 10.0, 10.0]);
    let result = Pelt::new(1, 1)
        .unwrap()
        .predict_penalty(&cost, 1_000.0)
        .unwrap();
    assert_eq!(result.breakpoints, [4]);
    assert_eq!(result.segment_cost, 100.0);
    assert_eq!(result.objective, 100.0);
}

#[test]
fn tied_objectives_use_lexicographically_smallest_partition() {
    let cost = l2(&[0.0, 1.0]);
    let result = Pelt::new(1, 1)
        .unwrap()
        .predict_penalty(&cost, 0.5)
        .unwrap();
    assert_eq!(result.breakpoints, [1, 2]);
    assert_eq!(result.objective, 0.5);
}

#[test]
fn matches_brute_force_and_unpruned_solver_on_small_problems() {
    for n_samples in 1..=10 {
        let signal: Vec<_> = (0..n_samples)
            .map(|index| ((index * 7 + n_samples * 3) % 11) as f64)
            .collect();
        let cost = l2(&signal);
        for min_size in 1..=n_samples.min(3) {
            for penalty in [0.1, 1.0, 5.0, 20.0, 100.0] {
                let expected = best_penalized(n_samples, min_size, penalty, |segment| {
                    cost.cost(segment).unwrap()
                })
                .unwrap();
                let grid = SearchGrid::new(min_size, 1).unwrap();
                let unpruned = solve_penalized(&cost, grid, penalty, false).unwrap();
                let pruned = solve_penalized(&cost, grid, penalty, true).unwrap();
                assert_eq!(
                    unpruned.breakpoints, expected.0,
                    "unpruned mismatch: n={n_samples}, min_size={min_size}, penalty={penalty}, signal={signal:?}"
                );
                assert!(objective_values_tied(unpruned.objective, expected.1));
                assert_eq!(
                    pruned.breakpoints, unpruned.breakpoints,
                    "pruning mismatch: n={n_samples}, min_size={min_size}, penalty={penalty}, signal={signal:?}"
                );
                assert!(objective_values_tied(pruned.objective, unpruned.objective));
            }
        }
    }
}

#[test]
fn pruning_and_unpruned_solver_match_on_approximate_grids() {
    for n_samples in 2..=20 {
        let signal: Vec<_> = (0..n_samples)
            .map(|index| ((index * 13 + n_samples) % 17) as f64)
            .collect();
        let cost = l2(&signal);
        for jump in 2..=4 {
            let grid = SearchGrid::new(1, jump).unwrap();
            for penalty in [0.25, 3.0, 30.0] {
                let unpruned = solve_penalized(&cost, grid, penalty, false).unwrap();
                let pruned = solve_penalized(&cost, grid, penalty, true).unwrap();
                assert_eq!(pruned.breakpoints, unpruned.breakpoints);
                assert!(objective_values_tied(pruned.objective, unpruned.objective));
            }
        }
    }
}

#[test]
fn jump_restricts_internal_breakpoints_but_not_terminal_sample() {
    let cost = l2(&[0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0]);
    let exact = Pelt::new(1, 1)
        .unwrap()
        .predict_penalty(&cost, 1.0)
        .unwrap();
    let grid = Pelt::new(1, 2)
        .unwrap()
        .predict_penalty(&cost, 1.0)
        .unwrap();
    assert_eq!(exact.breakpoints, [3, 7]);
    assert_eq!(grid.breakpoints, [2, 4, 7]);
}

#[test]
fn rejects_invalid_penalty_and_unsupported_stopping_rules() {
    let cost = l2(&[0.0, 1.0]);
    let detector = Pelt::new(1, 1).unwrap();
    assert!(matches!(
        detector.predict_penalty(&cost, 0.0),
        Err(Error::InvalidPenalty { .. })
    ));
    assert!(matches!(
        detector.predict(&cost, Stop::Changes(1)),
        Err(Error::UnsupportedStoppingRule { .. })
    ));
}

struct CountingL2 {
    inner: CostL2,
    pruning_constant: f64,
    evaluated_segments: AtomicUsize,
}

impl SegmentCost for CountingL2 {
    fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }

    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    fn min_size(&self) -> usize {
        self.inner.min_size()
    }

    fn cost(&self, segment: Range<usize>) -> Result<f64, Error> {
        self.inner.cost(segment)
    }

    fn costs_ending_at(
        &self,
        starts: &[usize],
        end: usize,
        output: &mut Vec<f64>,
    ) -> Result<(), Error> {
        self.evaluated_segments
            .fetch_add(starts.len(), Ordering::Relaxed);
        self.inner.costs_ending_at(starts, end, output)
    }

    fn pelt_pruning_constant(&self) -> Option<f64> {
        Some(self.pruning_constant)
    }
}

#[test]
fn pelt_uses_a_declared_nonzero_pruning_constant() {
    let signal: Vec<f64> = [0.0, 8.0, -5.0, 4.0]
        .into_iter()
        .flat_map(|level| std::iter::repeat_n(level, 20))
        .collect();
    let fitted = l2(&signal);
    let zero = CountingL2 {
        inner: fitted.clone(),
        pruning_constant: 0.0,
        evaluated_segments: AtomicUsize::new(0),
    };
    let conservative = CountingL2 {
        inner: fitted,
        // If K=0 is valid, every smaller K is also valid but prunes less.
        pruning_constant: -1.0e9,
        evaluated_segments: AtomicUsize::new(0),
    };
    let detector = Pelt::new(2, 1).unwrap();
    let zero_result = detector.predict_penalty(&zero, 1.0).unwrap();
    let conservative_result = detector.predict_penalty(&conservative, 1.0).unwrap();

    assert_eq!(zero_result.breakpoints, conservative_result.breakpoints);
    assert!(
        zero.evaluated_segments.load(Ordering::Relaxed)
            < conservative.evaluated_segments.load(Ordering::Relaxed)
    );
}
