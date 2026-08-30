use crate::{
    objective_values_tied, validate_penalty, Detector, DetectorCapabilities, Error, SearchGrid,
    SegmentCost, Segmentation, Stop,
};

use super::candidate_positions;

const NO_PREDECESSOR: usize = usize::MAX;

/// Exact penalized change-point detection using PELT pruning.
///
/// The pruning constant is zero, which is valid for the L2 segment cost and for
/// other costs whose value cannot increase when a segment is split. The result is
/// exact on the selected search grid when that pruning condition holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pelt {
    grid: SearchGrid,
}

impl Pelt {
    pub fn new(min_size: usize, jump: usize) -> Result<Self, Error> {
        Ok(Self {
            grid: SearchGrid::new(min_size, jump)?,
        })
    }

    pub fn grid(self) -> SearchGrid {
        self.grid
    }

    pub fn predict_penalty<C: SegmentCost>(
        &self,
        cost: &C,
        penalty: f64,
    ) -> Result<Segmentation, Error> {
        solve_penalized(cost, self.grid, penalty, true)
    }
}

impl<C: SegmentCost> Detector<C> for Pelt {
    fn capabilities(&self) -> DetectorCapabilities {
        DetectorCapabilities::PENALTY_ONLY
    }

    fn predict(&self, cost: &C, stop: Stop) -> Result<Segmentation, Error> {
        match stop {
            Stop::Penalty(penalty) => self.predict_penalty(cost, penalty),
            Stop::Changes(_) => Err(Error::UnsupportedStoppingRule {
                detector: "Pelt",
                rule: "changes",
            }),
            Stop::Budget(_) => Err(Error::UnsupportedStoppingRule {
                detector: "Pelt",
                rule: "budget",
            }),
        }
    }
}

fn solve_penalized<C: SegmentCost>(
    cost: &C,
    grid: SearchGrid,
    penalty: f64,
    pruning: bool,
) -> Result<Segmentation, Error> {
    validate_penalty(penalty)?;
    if grid.min_size < cost.min_size() {
        return Err(Error::MinSizeBelowCost {
            requested: grid.min_size,
            minimum: cost.min_size(),
        });
    }
    let min_size = grid.min_size;
    let n_samples = cost.n_samples();
    if n_samples < min_size {
        return Err(Error::SegmentTooShort {
            start: 0,
            end: n_samples,
            length: n_samples,
            minimum: min_size,
        });
    }

    let positions = candidate_positions(n_samples, grid.jump);
    let n_positions = positions.len();
    let mut best_cost = vec![f64::INFINITY; n_positions];
    let mut predecessor = vec![NO_PREDECESSOR; n_positions];
    let mut left_path = Vec::new();
    let mut right_path = Vec::new();
    best_cost[0] = -penalty;

    if pruning {
        solve_pruned(
            cost,
            min_size,
            penalty,
            &positions,
            &mut best_cost,
            &mut predecessor,
            &mut left_path,
            &mut right_path,
        )?;
    } else {
        solve_unpruned(
            cost,
            min_size,
            penalty,
            &positions,
            &mut best_cost,
            &mut predecessor,
            &mut left_path,
            &mut right_path,
        )?;
    }

    finish_segmentation(
        cost,
        min_size,
        penalty,
        &positions,
        &best_cost,
        &predecessor,
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_unpruned<C: SegmentCost>(
    cost: &C,
    min_size: usize,
    penalty: f64,
    positions: &[usize],
    best_cost: &mut [f64],
    predecessor: &mut [usize],
    left_path: &mut Vec<usize>,
    right_path: &mut Vec<usize>,
) -> Result<(), Error> {
    for end_index in 1..positions.len() {
        let end = positions[end_index];
        for start_index in 0..end_index {
            let start = positions[start_index];
            if !best_cost[start_index].is_finite() || end - start < min_size {
                continue;
            }
            let segment_cost = cost.cost(start..end)?;
            consider_candidate(
                start_index,
                end_index,
                best_cost[start_index] + segment_cost + penalty,
                positions,
                best_cost,
                predecessor,
                left_path,
                right_path,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn solve_pruned<C: SegmentCost>(
    cost: &C,
    min_size: usize,
    penalty: f64,
    positions: &[usize],
    best_cost: &mut [f64],
    predecessor: &mut [usize],
    left_path: &mut Vec<usize>,
    right_path: &mut Vec<usize>,
) -> Result<(), Error> {
    let mut active = Vec::with_capacity(positions.len());
    let mut evaluated = Vec::with_capacity(positions.len());
    let mut prune_at = vec![usize::MAX; positions.len()];
    active.push(0);

    for end_index in 1..positions.len() {
        let end = positions[end_index];
        active.retain(|&start_index| prune_at[start_index] > end);
        evaluated.clear();

        for &start_index in &active {
            let start = positions[start_index];
            if end - start < min_size {
                continue;
            }
            let segment_cost = cost.cost(start..end)?;
            evaluated.push((start_index, segment_cost));
            consider_candidate(
                start_index,
                end_index,
                best_cost[start_index] + segment_cost + penalty,
                positions,
                best_cost,
                predecessor,
                left_path,
                right_path,
            )?;
        }

        if best_cost[end_index].is_finite() {
            for &(start_index, segment_cost) in &evaluated {
                let pruning_score = best_cost[start_index] + segment_cost;
                if pruning_score > best_cost[end_index]
                    && !objective_values_tied(pruning_score, best_cost[end_index])
                {
                    // The proof through `end` only dominates this candidate once
                    // a following segment starting at `end` can satisfy min_size.
                    prune_at[start_index] = prune_at[start_index].min(end.saturating_add(min_size));
                }
            }
            active.push(end_index);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn consider_candidate(
    start_index: usize,
    end_index: usize,
    candidate_cost: f64,
    positions: &[usize],
    best_cost: &mut [f64],
    predecessor: &mut [usize],
    left_path: &mut Vec<usize>,
    right_path: &mut Vec<usize>,
) -> Result<(), Error> {
    if !candidate_cost.is_finite() {
        return Err(Error::NonFiniteObjective {
            value: candidate_cost,
        });
    }

    let best_predecessor = predecessor[end_index];
    let better = if !best_cost[end_index].is_finite() {
        true
    } else if objective_values_tied(candidate_cost, best_cost[end_index]) {
        candidate_path_is_smaller(
            start_index,
            best_predecessor,
            end_index,
            positions,
            predecessor,
            left_path,
            right_path,
        )?
    } else {
        candidate_cost < best_cost[end_index]
    };

    if better {
        best_cost[end_index] = candidate_cost;
        predecessor[end_index] = start_index;
    }
    Ok(())
}

fn candidate_path_is_smaller(
    candidate_start: usize,
    best_start: usize,
    end_index: usize,
    positions: &[usize],
    predecessor: &[usize],
    candidate_path: &mut Vec<usize>,
    best_path: &mut Vec<usize>,
) -> Result<bool, Error> {
    if best_start == NO_PREDECESSOR {
        return Ok(true);
    }
    build_candidate_path(
        candidate_start,
        end_index,
        positions,
        predecessor,
        candidate_path,
    )?;
    build_candidate_path(best_start, end_index, positions, predecessor, best_path)?;
    Ok(candidate_path < best_path)
}

fn build_candidate_path(
    start_index: usize,
    end_index: usize,
    positions: &[usize],
    predecessor: &[usize],
    output: &mut Vec<usize>,
) -> Result<(), Error> {
    output.clear();
    let mut cursor = start_index;
    while cursor != 0 {
        output.push(positions[cursor]);
        cursor = predecessor[cursor];
        if cursor == NO_PREDECESSOR {
            return Err(Error::NumericalFailure {
                context: "comparing penalized segmentation paths",
            });
        }
    }
    output.reverse();
    output.push(positions[end_index]);
    Ok(())
}

fn finish_segmentation<C: SegmentCost>(
    cost: &C,
    min_size: usize,
    penalty: f64,
    positions: &[usize],
    best_cost: &[f64],
    predecessor: &[usize],
) -> Result<Segmentation, Error> {
    let final_index = positions.len() - 1;
    if !best_cost[final_index].is_finite() {
        return Err(Error::NumericalFailure {
            context: "finding a feasible penalized segmentation",
        });
    }

    let mut breakpoints = Vec::new();
    let mut cursor = final_index;
    while cursor != 0 {
        breakpoints.push(positions[cursor]);
        cursor = predecessor[cursor];
        if cursor == NO_PREDECESSOR {
            return Err(Error::NumericalFailure {
                context: "backtracking a penalized segmentation",
            });
        }
    }
    breakpoints.reverse();

    let mut start = 0;
    let mut segment_cost = 0.0;
    for &end in &breakpoints {
        segment_cost += cost.cost(start..end)?;
        start = end;
    }
    let objective = segment_cost + penalty * (breakpoints.len() - 1) as f64;
    Segmentation::new(
        breakpoints,
        segment_cost,
        objective,
        cost.n_samples(),
        min_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::best_penalized;
    use crate::{CostL2, SignalView};

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
}
