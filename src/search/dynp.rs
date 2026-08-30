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

        let positions = candidate_positions(cost.n_samples(), self.grid.jump);
        let n_positions = positions.len();
        let n_segments = changes
            .checked_add(1)
            .ok_or_else(|| self.infeasible(cost.n_samples(), changes, min_size))?;
        let predecessor_len = n_segments
            .checked_add(1)
            .and_then(|rows| rows.checked_mul(n_positions))
            .ok_or(Error::NumericalFailure {
                context: "allocating Dynp predecessor table",
            })?;

        let mut predecessors = vec![NO_PREDECESSOR; predecessor_len];
        let mut previous_scores = vec![f64::INFINITY; n_positions];
        let mut current_scores = vec![f64::INFINITY; n_positions];
        let mut previous_ranks = vec![NO_PREDECESSOR; n_positions];
        let mut current_ranks = vec![NO_PREDECESSOR; n_positions];
        previous_scores[0] = 0.0;
        previous_ranks[0] = 0;

        for segment_count in 1..=n_segments {
            current_scores.fill(f64::INFINITY);
            current_ranks.fill(NO_PREDECESSOR);

            for end_index in 1..n_positions {
                let end = positions[end_index];
                let mut best_predecessor = NO_PREDECESSOR;
                let mut best_score = f64::INFINITY;

                for start_index in 0..end_index {
                    let start = positions[start_index];
                    if previous_ranks[start_index] == NO_PREDECESSOR || end - start < min_size {
                        continue;
                    }
                    let segment_cost = cost.cost(start..end)?;
                    let candidate_score = previous_scores[start_index] + segment_cost;
                    if !candidate_score.is_finite() {
                        return Err(Error::NonFiniteObjective {
                            value: candidate_score,
                        });
                    }

                    let best_rank = if best_predecessor == NO_PREDECESSOR {
                        NO_PREDECESSOR
                    } else {
                        previous_ranks[best_predecessor]
                    };
                    if score_is_better(
                        candidate_score,
                        previous_ranks[start_index],
                        best_score,
                        best_rank,
                    ) {
                        best_score = candidate_score;
                        best_predecessor = start_index;
                    }
                }

                if best_predecessor != NO_PREDECESSOR {
                    current_scores[end_index] = best_score;
                    predecessors[segment_count * n_positions + end_index] = best_predecessor;
                }
            }

            assign_lexicographic_ranks(
                &positions,
                &predecessors[segment_count * n_positions..(segment_count + 1) * n_positions],
                &previous_ranks,
                &mut current_ranks,
            );
            std::mem::swap(&mut previous_scores, &mut current_scores);
            std::mem::swap(&mut previous_ranks, &mut current_ranks);
        }

        let final_index = n_positions - 1;
        let final_score = previous_scores[final_index];
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

fn score_is_better(
    candidate_score: f64,
    candidate_rank: usize,
    best_score: f64,
    best_rank: usize,
) -> bool {
    if !best_score.is_finite() {
        return true;
    }
    if objective_values_tied(candidate_score, best_score) {
        candidate_rank < best_rank
    } else {
        candidate_score < best_score
    }
}

fn assign_lexicographic_ranks(
    positions: &[usize],
    predecessors: &[usize],
    previous_ranks: &[usize],
    output: &mut [usize],
) {
    let mut reachable: Vec<_> = predecessors
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, predecessor)| *predecessor != NO_PREDECESSOR)
        .map(|(end_index, predecessor)| {
            (previous_ranks[predecessor], positions[end_index], end_index)
        })
        .collect();
    reachable.sort_unstable();
    for (rank, (_, _, end_index)) in reachable.into_iter().enumerate() {
        output[end_index] = rank;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::best_fixed_changes;
    use crate::{CostL2, SignalView};

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
}
