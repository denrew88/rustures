use crate::{
    objective_values_tied, Detector, DetectorCapabilities, Error, SearchGrid, SegmentCost,
    Segmentation, Stop,
};

use super::candidate_positions;

const NO_PREDECESSOR: usize = usize::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dynp {
    grid: SearchGrid,
}

impl Dynp {
    pub fn new(min_size: usize, jump: usize) -> Result<Self, Error> {
        Ok(Self {
            grid: SearchGrid::new(min_size, jump)?,
        })
    }

    pub fn grid(self) -> SearchGrid {
        self.grid
    }

    pub fn predict_changes<C: SegmentCost>(
        &self,
        cost: &C,
        changes: usize,
    ) -> Result<Segmentation, Error> {
        let min_size = self.effective_min_size(cost)?;
        validate_feasible(cost.n_samples(), changes, min_size, self.grid.jump)?;
        if changes == 0 {
            let segment_cost = cost.cost(0..cost.n_samples())?;
            return Segmentation::new(
                vec![cost.n_samples()],
                segment_cost,
                segment_cost,
                cost.n_samples(),
                min_size,
            );
        }

        let positions = candidate_positions(cost.n_samples(), self.grid.jump);
        let n_positions = positions.len();
        let n_segments = changes
            .checked_add(1)
            .ok_or_else(|| self.infeasible(cost.n_samples(), changes, min_size))?;
        let state_len = n_segments
            .checked_add(1)
            .and_then(|rows| rows.checked_mul(n_positions))
            .ok_or(Error::NumericalFailure {
                context: "allocating Dynp state tables",
            })?;

        let mut predecessors = vec![NO_PREDECESSOR; state_len];
        let mut scores = vec![f64::INFINITY; state_len];
        let mut segment_costs = Vec::new();
        let mut candidate_path = Vec::new();
        let mut best_path = Vec::new();
        scores[0] = 0.0;

        // Endpoint-major traversal evaluates every admissible segment cost once
        // and reuses that batch for all fixed-K states ending at this position.
        for end_index in 1..n_positions {
            let end = positions[end_index];
            let valid_start_count =
                positions[..end_index].partition_point(|&start| end - start >= min_size);
            if valid_start_count == 0 {
                continue;
            }
            cost.costs_ending_at(&positions[..valid_start_count], end, &mut segment_costs)?;
            if segment_costs.len() != valid_start_count {
                return Err(Error::NumericalFailure {
                    context: "evaluating a Dynp endpoint cost batch",
                });
            }
            if let Some(value) = segment_costs.iter().find(|value| !value.is_finite()) {
                return Err(Error::NonFiniteObjective { value: *value });
            }

            for segment_count in 1..=n_segments.min(end_index) {
                let mut best_predecessor = NO_PREDECESSOR;
                let mut best_score = f64::INFINITY;

                let previous_row = (segment_count - 1) * n_positions;
                for (start_index, &segment_cost) in
                    segment_costs[..valid_start_count].iter().enumerate()
                {
                    let previous_score = scores[previous_row + start_index];
                    if !previous_score.is_finite() {
                        continue;
                    }
                    let candidate_score = previous_score + segment_cost;
                    if !candidate_score.is_finite() {
                        return Err(Error::NonFiniteObjective {
                            value: candidate_score,
                        });
                    }

                    let better = if !best_score.is_finite() {
                        true
                    } else if objective_values_tied(candidate_score, best_score) {
                        state_path_is_smaller(
                            segment_count - 1,
                            start_index,
                            best_predecessor,
                            &positions,
                            &predecessors,
                            n_positions,
                            &mut candidate_path,
                            &mut best_path,
                        )?
                    } else {
                        candidate_score < best_score
                    };
                    if better {
                        best_score = candidate_score;
                        best_predecessor = start_index;
                    }
                }

                if best_predecessor != NO_PREDECESSOR {
                    scores[segment_count * n_positions + end_index] = best_score;
                    predecessors[segment_count * n_positions + end_index] = best_predecessor;
                }
            }
        }

        let final_index = n_positions - 1;
        let final_score = scores[n_segments * n_positions + final_index];
        if !final_score.is_finite() {
            return Err(self.infeasible(cost.n_samples(), changes, min_size));
        }

        let mut breakpoints = Vec::with_capacity(n_segments);
        let mut end_index = final_index;
        for segment_count in (1..=n_segments).rev() {
            breakpoints.push(positions[end_index]);
            end_index = predecessors[segment_count * n_positions + end_index];
            if end_index == NO_PREDECESSOR {
                return Err(Error::NumericalFailure {
                    context: "backtracking a Dynp segmentation",
                });
            }
        }
        if end_index != 0 {
            return Err(Error::NumericalFailure {
                context: "backtracking a Dynp segmentation",
            });
        }
        breakpoints.reverse();

        Segmentation::new(
            breakpoints,
            final_score,
            final_score,
            cost.n_samples(),
            min_size,
        )
    }

    fn effective_min_size<C: SegmentCost>(&self, cost: &C) -> Result<usize, Error> {
        if self.grid.min_size < cost.min_size() {
            return Err(Error::MinSizeBelowCost {
                requested: self.grid.min_size,
                minimum: cost.min_size(),
            });
        }
        Ok(self.grid.min_size)
    }

    fn infeasible(&self, n_samples: usize, changes: usize, min_size: usize) -> Error {
        Error::InfeasibleSegmentation {
            n_samples,
            changes,
            min_size,
            jump: self.grid.jump,
        }
    }
}

impl<C: SegmentCost> Detector<C> for Dynp {
    fn capabilities(&self) -> DetectorCapabilities {
        DetectorCapabilities::CHANGES_ONLY
    }

    fn predict(&self, cost: &C, stop: Stop) -> Result<Segmentation, Error> {
        match stop {
            Stop::Changes(changes) => self.predict_changes(cost, changes),
            Stop::Penalty(_) => Err(Error::UnsupportedStoppingRule {
                detector: "Dynp",
                rule: "penalty",
            }),
            Stop::Budget(_) => Err(Error::UnsupportedStoppingRule {
                detector: "Dynp",
                rule: "budget",
            }),
        }
    }
}

fn validate_feasible(
    n_samples: usize,
    changes: usize,
    min_size: usize,
    jump: usize,
) -> Result<(), Error> {
    let infeasible = || Error::InfeasibleSegmentation {
        n_samples,
        changes,
        min_size,
        jump,
    };
    if n_samples < min_size {
        return Err(infeasible());
    }
    if changes == 0 {
        return Ok(());
    }

    let grid_steps = min_size.div_ceil(jump);
    let internal_spacing = grid_steps.checked_mul(jump).ok_or_else(infeasible)?;
    let required_internal = changes
        .checked_mul(internal_spacing)
        .ok_or_else(infeasible)?;
    let latest_internal = n_samples - min_size;
    if required_internal > latest_internal {
        return Err(infeasible());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn state_path_is_smaller(
    segment_count: usize,
    candidate_end: usize,
    best_end: usize,
    positions: &[usize],
    predecessors: &[usize],
    n_positions: usize,
    candidate_path: &mut Vec<usize>,
    best_path: &mut Vec<usize>,
) -> Result<bool, Error> {
    if best_end == NO_PREDECESSOR {
        return Ok(true);
    }
    build_state_path(
        segment_count,
        candidate_end,
        positions,
        predecessors,
        n_positions,
        candidate_path,
    )?;
    build_state_path(
        segment_count,
        best_end,
        positions,
        predecessors,
        n_positions,
        best_path,
    )?;
    Ok(candidate_path < best_path)
}

fn build_state_path(
    segment_count: usize,
    end_index: usize,
    positions: &[usize],
    predecessors: &[usize],
    n_positions: usize,
    output: &mut Vec<usize>,
) -> Result<(), Error> {
    output.clear();
    let mut cursor = end_index;
    for count in (1..=segment_count).rev() {
        output.push(positions[cursor]);
        cursor = predecessors[count * n_positions + cursor];
        if cursor == NO_PREDECESSOR {
            return Err(Error::NumericalFailure {
                context: "comparing Dynp segmentation paths",
            });
        }
    }
    if cursor != 0 {
        return Err(Error::NumericalFailure {
            context: "comparing Dynp segmentation paths",
        });
    }
    output.reverse();
    Ok(())
}

#[cfg(test)]
mod tests {
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
}
