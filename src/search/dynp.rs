use crate::{
    objective_values_tied, Detector, DetectorCapabilities, Error, SearchGrid, SegmentCost,
    Segmentation, Stop,
};

pub const DEFAULT_DYNP_MEMORY_LIMIT_BYTES: usize = 536_870_912;
const NO_PREDECESSOR: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dynp {
    grid: SearchGrid,
    max_memory_bytes: usize,
}

impl Dynp {
    pub fn new(min_size: usize, jump: usize) -> Result<Self, Error> {
        Self::with_memory_limit(min_size, jump, DEFAULT_DYNP_MEMORY_LIMIT_BYTES)
    }

    pub fn with_memory_limit(
        min_size: usize,
        jump: usize,
        max_memory_bytes: usize,
    ) -> Result<Self, Error> {
        if max_memory_bytes == 0 {
            return Err(Error::InvalidMemoryLimit {
                value: max_memory_bytes,
            });
        }
        Ok(Self {
            grid: SearchGrid::new(min_size, jump)?,
            max_memory_bytes,
        })
    }

    pub fn grid(self) -> SearchGrid {
        self.grid
    }

    pub fn max_memory_bytes(self) -> usize {
        self.max_memory_bytes
    }

    pub fn estimated_memory_bytes(&self, n_samples: usize, changes: usize) -> Result<usize, Error> {
        if changes == 0 {
            return Ok(0);
        }
        let n_positions = candidate_position_count(n_samples, self.grid.jump)?;
        if n_positions >= NO_PREDECESSOR as usize {
            return Err(Error::DynpMemoryLimit {
                requested: usize::MAX,
                maximum: self.max_memory_bytes,
            });
        }
        let n_segments = changes.checked_add(1).ok_or(Error::DynpMemoryLimit {
            requested: usize::MAX,
            maximum: self.max_memory_bytes,
        })?;
        let state_len = n_segments
            .checked_add(1)
            .and_then(|rows| rows.checked_mul(n_positions))
            .ok_or(Error::DynpMemoryLimit {
                requested: usize::MAX,
                maximum: self.max_memory_bytes,
            })?;

        let state_bytes = state_len
            .checked_mul(size_of::<f64>() + size_of::<u32>())
            .ok_or(Error::DynpMemoryLimit {
                requested: usize::MAX,
                maximum: self.max_memory_bytes,
            })?;
        let position_and_batch_bytes = n_positions
            .checked_mul(size_of::<usize>() + size_of::<f64>())
            .ok_or(Error::DynpMemoryLimit {
                requested: usize::MAX,
                maximum: self.max_memory_bytes,
            })?;
        let path_bytes =
            n_segments
                .checked_mul(3 * size_of::<usize>())
                .ok_or(Error::DynpMemoryLimit {
                    requested: usize::MAX,
                    maximum: self.max_memory_bytes,
                })?;
        state_bytes
            .checked_add(position_and_batch_bytes)
            .and_then(|bytes| bytes.checked_add(path_bytes))
            .ok_or(Error::DynpMemoryLimit {
                requested: usize::MAX,
                maximum: self.max_memory_bytes,
            })
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

        let estimated_memory = self.estimated_memory_bytes(cost.n_samples(), changes)?;
        if estimated_memory > self.max_memory_bytes {
            return Err(Error::DynpMemoryLimit {
                requested: estimated_memory,
                maximum: self.max_memory_bytes,
            });
        }

        let positions = checked_candidate_positions(cost.n_samples(), self.grid.jump)?;
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

        let mut predecessors = try_filled_vec(
            state_len,
            NO_PREDECESSOR,
            "allocating Dynp predecessor states",
        )?;
        let mut scores = try_filled_vec(state_len, f64::INFINITY, "allocating Dynp score states")?;
        let mut segment_costs = Vec::new();
        segment_costs
            .try_reserve_exact(n_positions)
            .map_err(|_| Error::AllocationFailure {
                context: "allocating a Dynp endpoint cost batch",
            })?;
        let mut candidate_path = Vec::new();
        let mut best_path = Vec::new();
        candidate_path
            .try_reserve_exact(n_segments)
            .map_err(|_| Error::AllocationFailure {
                context: "allocating a Dynp tie path",
            })?;
        best_path
            .try_reserve_exact(n_segments)
            .map_err(|_| Error::AllocationFailure {
                context: "allocating a Dynp tie path",
            })?;
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
                let mut best_predecessor = usize::MAX;
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

                if best_predecessor != usize::MAX {
                    scores[segment_count * n_positions + end_index] = best_score;
                    predecessors[segment_count * n_positions + end_index] =
                        u32::try_from(best_predecessor).map_err(|_| Error::NumericalFailure {
                            context: "storing a compact Dynp predecessor",
                        })?;
                }
            }
        }

        let final_index = n_positions - 1;
        let final_score = scores[n_segments * n_positions + final_index];
        if !final_score.is_finite() {
            return Err(self.infeasible(cost.n_samples(), changes, min_size));
        }

        let mut breakpoints = Vec::new();
        breakpoints
            .try_reserve_exact(n_segments)
            .map_err(|_| Error::AllocationFailure {
                context: "allocating Dynp breakpoints",
            })?;
        let mut end_index = final_index;
        for segment_count in (1..=n_segments).rev() {
            breakpoints.push(positions[end_index]);
            let predecessor = predecessors[segment_count * n_positions + end_index];
            if predecessor == NO_PREDECESSOR {
                return Err(Error::NumericalFailure {
                    context: "backtracking a Dynp segmentation",
                });
            }
            end_index = predecessor as usize;
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

fn candidate_position_count(n_samples: usize, jump: usize) -> Result<usize, Error> {
    if n_samples == 0 {
        return Err(Error::EmptySignal);
    }
    if jump == 0 {
        return Err(Error::InvalidJump { value: jump });
    }
    (n_samples - 1)
        .checked_div(jump)
        .and_then(|internal| internal.checked_add(2))
        .ok_or(Error::NumericalFailure {
            context: "counting Dynp search-grid positions",
        })
}

fn checked_candidate_positions(n_samples: usize, jump: usize) -> Result<Vec<usize>, Error> {
    let expected = candidate_position_count(n_samples, jump)?;
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(expected)
        .map_err(|_| Error::AllocationFailure {
            context: "allocating Dynp search-grid positions",
        })?;
    positions.push(0);
    let mut position = jump;
    while position < n_samples {
        positions.push(position);
        match position.checked_add(jump) {
            Some(next) => position = next,
            None => break,
        }
    }
    positions.push(n_samples);
    debug_assert_eq!(positions.len(), expected);
    Ok(positions)
}

fn try_filled_vec<T: Clone>(
    length: usize,
    value: T,
    context: &'static str,
) -> Result<Vec<T>, Error> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| Error::AllocationFailure { context })?;
    output.resize(length, value);
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn state_path_is_smaller(
    segment_count: usize,
    candidate_end: usize,
    best_end: usize,
    positions: &[usize],
    predecessors: &[u32],
    n_positions: usize,
    candidate_path: &mut Vec<usize>,
    best_path: &mut Vec<usize>,
) -> Result<bool, Error> {
    if best_end == usize::MAX {
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
    predecessors: &[u32],
    n_positions: usize,
    output: &mut Vec<usize>,
) -> Result<(), Error> {
    output.clear();
    let mut cursor = end_index;
    for count in (1..=segment_count).rev() {
        output.push(positions[cursor]);
        let predecessor = predecessors[count * n_positions + cursor];
        if predecessor == NO_PREDECESSOR {
            return Err(Error::NumericalFailure {
                context: "comparing Dynp segmentation paths",
            });
        }
        cursor = predecessor as usize;
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
#[path = "../../tests/unit/search/dynp.rs"]
mod tests;
