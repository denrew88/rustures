use super::*;
use crate::oracle::best_fixed_changes;
use crate::{CostL2, SignalView};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

fn l2(signal: &[f64]) -> CostL2 {
    CostL2::fit(SignalView::new(signal, signal.len(), 1).unwrap()).unwrap()
}

#[test]
fn finds_known_two_change_partition() {
    let cost = l2(&[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, -5.0, -5.0, -5.0]);
    let result = Dynp::new(2, 1).unwrap().predict_changes(&cost, 2).unwrap();
    assert_eq!(result.breakpoints, [3, 6, 9]);
    assert_eq!(result.segment_cost, 0.0);
}

#[test]
fn constant_signal_uses_lexicographically_smallest_breakpoints() {
    let cost = l2(&[4.0; 8]);
    let result = Dynp::new(1, 1).unwrap().predict_changes(&cost, 3).unwrap();
    assert_eq!(result.breakpoints, [1, 2, 3, 8]);
}

#[test]
fn jump_restricts_internal_breakpoints_but_not_terminal_sample() {
    let cost = l2(&[0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0]);
    let result = Dynp::new(1, 3).unwrap().predict_changes(&cost, 1).unwrap();
    assert_eq!(result.breakpoints, [3, 7]);
}

#[test]
fn rejects_infeasible_problem_before_dp() {
    let cost = l2(&[0.0; 5]);
    assert_eq!(
        Dynp::new(2, 1).unwrap().predict_changes(&cost, 2),
        Err(Error::InfeasibleSegmentation {
            n_samples: 5,
            changes: 2,
            min_size: 2,
            jump: 1,
        })
    );
}

#[test]
fn matches_brute_force_for_all_small_fixed_change_problems() {
    for n_samples in 1..=10 {
        let signal: Vec<_> = (0..n_samples)
            .map(|index| ((index * 7 + n_samples * 3) % 11) as f64)
            .collect();
        let cost = l2(&signal);
        for min_size in 1..=n_samples.min(3) {
            let maximum_changes = n_samples / min_size - 1;
            for changes in 0..=maximum_changes {
                let expected = best_fixed_changes(n_samples, min_size, changes, |segment| {
                    cost.cost(segment).unwrap()
                })
                .unwrap();
                let expected = expected.unwrap();
                let actual = Dynp::new(min_size, 1)
                    .unwrap()
                    .predict_changes(&cost, changes)
                    .unwrap();
                assert_eq!(
                    actual.breakpoints, expected.0,
                    "n_samples={n_samples}, min_size={min_size}, changes={changes}, actual_cost={}, expected_cost={}",
                    actual.segment_cost, expected.1
                );
                assert_eq!(actual.segment_cost, expected.1);
            }
        }
    }
}

#[test]
fn reports_only_supported_stopping_rule() {
    let detector = Dynp::new(1, 1).unwrap();
    let cost = l2(&[0.0, 1.0]);
    assert_eq!(
        detector.predict(&cost, Stop::Penalty(1.0)),
        Err(Error::UnsupportedStoppingRule {
            detector: "Dynp",
            rule: "penalty",
        })
    );
}

struct CountingBatchCost {
    values: Vec<f64>,
    batch_calls: AtomicUsize,
    scalar_calls: AtomicUsize,
}

impl SegmentCost for CountingBatchCost {
    fn n_samples(&self) -> usize {
        self.values.len()
    }
    fn n_features(&self) -> usize {
        1
    }
    fn min_size(&self) -> usize {
        1
    }
    fn cost(&self, segment: Range<usize>) -> Result<f64, Error> {
        self.scalar_calls.fetch_add(1, Ordering::Relaxed);
        let values = &self.values[segment];
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        Ok(values.iter().map(|value| (value - mean).powi(2)).sum())
    }
    fn costs_ending_at(
        &self,
        starts: &[usize],
        end: usize,
        output: &mut Vec<f64>,
    ) -> Result<(), Error> {
        self.batch_calls.fetch_add(1, Ordering::Relaxed);
        output.clear();
        for &start in starts {
            let values = &self.values[start..end];
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            output.push(values.iter().map(|value| (value - mean).powi(2)).sum());
        }
        Ok(())
    }
}

#[test]
fn endpoint_cost_batch_is_shared_across_all_k_states() {
    let cost = CountingBatchCost {
        values: vec![0.0, 0.0, 3.0, 3.0, -2.0, -2.0, 1.0, 1.0],
        batch_calls: AtomicUsize::new(0),
        scalar_calls: AtomicUsize::new(0),
    };
    let result = Dynp::new(1, 1).unwrap().predict_changes(&cost, 3).unwrap();
    assert_eq!(result.breakpoints, [2, 4, 6, 8]);
    assert_eq!(cost.batch_calls.load(Ordering::Relaxed), 8);
    assert_eq!(cost.scalar_calls.load(Ordering::Relaxed), 0);
}

#[test]
fn estimates_compact_state_memory_and_rejects_before_cost_evaluation() {
    let cost = CountingBatchCost {
        values: vec![0.0, 0.0, 3.0, 3.0, -2.0, -2.0, 1.0, 1.0],
        batch_calls: AtomicUsize::new(0),
        scalar_calls: AtomicUsize::new(0),
    };
    let detector = Dynp::with_memory_limit(1, 1, 779).unwrap();
    // M=9 positions, K+2=5 state rows: 45*(8+4) state bytes,
    // 9*(8+8) endpoint bytes, and 3*(K+1)*8 path bytes.
    assert_eq!(detector.estimated_memory_bytes(8, 3), Ok(780));
    assert_eq!(
        detector.predict_changes(&cost, 3),
        Err(Error::DynpMemoryLimit {
            requested: 780,
            maximum: 779,
        })
    );
    assert_eq!(cost.batch_calls.load(Ordering::Relaxed), 0);
    assert_eq!(cost.scalar_calls.load(Ordering::Relaxed), 0);
}

#[test]
fn memory_estimate_respects_jump_and_zero_change_fast_path() {
    let detector = Dynp::new(1, 3).unwrap();
    assert_eq!(detector.estimated_memory_bytes(7, 0), Ok(0));
    // Positions are [0, 3, 6, 7], so M=4 and K=1 gives 3 rows.
    assert_eq!(detector.estimated_memory_bytes(7, 1), Ok(256));
    assert_eq!(
        Dynp::with_memory_limit(1, 1, 0),
        Err(Error::InvalidMemoryLimit { value: 0 })
    );
}
