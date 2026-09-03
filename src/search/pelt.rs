use crate::{
    objective_values_tied, validate_penalty, Detector, DetectorCapabilities, Error, SearchGrid,
    SegmentCost, Segmentation, Stop,
};

use super::candidate_positions;

const NO_PREDECESSOR: usize = usize::MAX;

/// Exact penalized change-point detection using PELT pruning.
///
/// A cost opts into pruning by returning a constant that satisfies the PELT
/// segment-combination inequality. The common L2 constant is zero. Costs that
/// return `None` use exact unpruned optimal partitioning on the selected grid.
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
        solve_penalized(
            cost,
            self.grid,
            penalty,
            cost.pelt_pruning_constant().is_some(),
        )
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
    let mut segment_costs = Vec::new();
    for end_index in 1..positions.len() {
        let end = positions[end_index];
        let valid_start_count =
            positions[..end_index].partition_point(|&start| end - start >= min_size);
        cost.costs_ending_at(&positions[..valid_start_count], end, &mut segment_costs)?;
        if segment_costs.len() != valid_start_count {
            return Err(Error::NumericalFailure {
                context: "evaluating an unpruned Pelt endpoint cost batch",
            });
        }
        for (start_index, &segment_cost) in segment_costs.iter().enumerate() {
            if !best_cost[start_index].is_finite() {
                continue;
            }
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
    let pruning_constant = cost.pelt_pruning_constant().unwrap_or(0.0);
    let mut active = Vec::with_capacity(positions.len());
    let mut evaluated = Vec::with_capacity(positions.len());
    let mut evaluated_starts = Vec::with_capacity(positions.len());
    let mut segment_costs = Vec::with_capacity(positions.len());
    let mut prune_at = vec![usize::MAX; positions.len()];
    active.push(0);

    for end_index in 1..positions.len() {
        let end = positions[end_index];
        active.retain(|&start_index| prune_at[start_index] > end);
        evaluated.clear();
        evaluated_starts.clear();

        for &start_index in &active {
            let start = positions[start_index];
            if end - start < min_size {
                continue;
            }
            evaluated_starts.push(start);
        }
        cost.costs_ending_at(&evaluated_starts, end, &mut segment_costs)?;
        if segment_costs.len() != evaluated_starts.len() {
            return Err(Error::NumericalFailure {
                context: "evaluating a pruned Pelt endpoint cost batch",
            });
        }

        let mut cost_index = 0;
        for &start_index in &active {
            let start = positions[start_index];
            if end - start < min_size {
                continue;
            }
            let segment_cost = segment_costs[cost_index];
            cost_index += 1;
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
                let pruning_score = best_cost[start_index] + segment_cost + pruning_constant;
                if !pruning_score.is_finite() {
                    return Err(Error::NonFiniteObjective {
                        value: pruning_score,
                    });
                }
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
#[path = "../../tests/unit/search/pelt.rs"]
mod tests;
