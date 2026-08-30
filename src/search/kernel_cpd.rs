use crate::core::segmentation::{
    non_negative_candidate_is_significant, non_negative_increase_is_significant,
    non_negative_score_tolerance,
};
use crate::{validate_penalty, Error, Kernel, SearchGrid, Segmentation, SignalView};

const NO_PREDECESSOR: usize = usize::MAX;

/// Exact kernel change-point detection with kernel accumulation and dynamic
/// programming fused into the same end/start scan.
#[derive(Clone, Debug)]
pub struct FusedKernelCPD<K> {
    values: Vec<f64>,
    n_samples: usize,
    n_features: usize,
    kernel: K,
    grid: SearchGrid,
}

impl<K: Kernel> FusedKernelCPD<K> {
    pub fn fit(
        signal: SignalView<'_>,
        kernel: K,
        min_size: usize,
        jump: usize,
    ) -> Result<Self, Error> {
        let shape = signal.shape();
        Ok(Self {
            values: signal.values().to_vec(),
            n_samples: shape.n_samples,
            n_features: shape.n_features,
            kernel,
            grid: SearchGrid::new(min_size, jump)?,
        })
    }

    pub fn kernel(&self) -> &K {
        &self.kernel
    }

    pub fn grid(&self) -> SearchGrid {
        self.grid
    }

    pub fn predict_changes(&self, changes: usize) -> Result<Segmentation, Error> {
        let segment_count = changes
            .checked_add(1)
            .ok_or_else(|| self.infeasible(changes))?;
        let spacing = self.validate_feasible(changes)?;
        // Store states by the number of breakpoints, as `0..=changes`.
        // The previous representation stored segment counts and needed an
        // unused extra column, making every row `changes + 2` wide.
        let width = segment_count;
        let table_len = (self.n_samples + 1)
            .checked_mul(width)
            .ok_or(Error::NumericalFailure {
                context: "allocating fused KernelCPD tables",
            })?;
        let mut scores = vec![f64::INFINITY; table_len];
        let mut predecessors = vec![NO_PREDECESSOR; table_len];
        let mut current_tolerances = vec![0.0; width];
        let mut diagonal_prefix = vec![0.0; self.n_samples + 1];
        let mut block_sums = vec![0.0; self.n_samples + 1];

        for end in 1..=self.n_samples {
            self.extend_kernel_sums(end, &mut diagonal_prefix, &mut block_sums);
            if !self.on_grid(end) || end < self.grid.min_size {
                continue;
            }
            let latest_start = end - self.grid.min_size;
            let end_diagonal = diagonal_prefix[end];
            let end_offset = end * width;
            let (previous_score_rows, current_and_later_scores) = scores.split_at_mut(end_offset);
            let current_scores = &mut current_and_later_scores[..width];
            let current_predecessors = &mut predecessors[end_offset..end_offset + width];

            // The no-change state does not depend on a previous DP row.
            current_scores[0] = kernel_segment_cost(end_diagonal, block_sums[0], end)?;
            if changes == 0 || latest_start < spacing {
                continue;
            }

            for start in (spacing..=latest_start).step_by(self.grid.jump) {
                debug_assert!(start < end);
                debug_assert!(start < diagonal_prefix.len());
                debug_assert!(start < block_sums.len());
                // SAFETY: `start <= latest_start < end <= n_samples`, and all
                // three accumulation arrays have `n_samples + 1` entries.
                let (diagonal_start, block_sum) = unsafe {
                    (
                        *diagonal_prefix.get_unchecked(start),
                        *block_sums.get_unchecked(start),
                    )
                };
                let segment_cost =
                    kernel_segment_cost(end_diagonal - diagonal_start, block_sum, end - start)?;
                let start_offset = start * width;
                debug_assert!(start_offset + width <= previous_score_rows.len());
                // SAFETY: `start < end`, so this complete row lies before the
                // current row produced by `split_at_mut(end * width)`.
                let previous_scores = unsafe {
                    previous_score_rows.get_unchecked(start_offset..start_offset + width)
                };
                let feasible_changes = start / spacing;
                let maximum_changes = changes.min(feasible_changes);
                let initializes_new_state = start == feasible_changes * spacing;
                debug_assert!(maximum_changes < width);
                for (current_changes, current_tolerance) in current_tolerances
                    .iter_mut()
                    .enumerate()
                    .take(maximum_changes + 1)
                    .skip(1)
                {
                    // Every prefix state in this range is structurally
                    // feasible. Its value is finite or +infinity after an
                    // arithmetic overflow; adding a finite non-negative
                    // segment cost preserves that invariant without a
                    // per-state `is_finite` branch.
                    // SAFETY: `current_changes <= changes < width`, so this
                    // state and the preceding state are within complete rows.
                    let previous = unsafe { *previous_scores.get_unchecked(current_changes - 1) };
                    let candidate = previous + segment_cost;
                    // SAFETY: the same state-width invariant above applies.
                    let current = unsafe { current_scores.get_unchecked_mut(current_changes) };
                    // `current_changes * spacing` is the earliest start that
                    // can follow a prefix with `current_changes` segments.
                    // Initialize that state directly, matching the compact C
                    // recurrence and removing the hot-loop infinity test.
                    if initializes_new_state && current_changes == feasible_changes {
                        *current = candidate;
                        // A non-finite candidate can only be +infinity after
                        // overflow. A zero cached tolerance lets a later
                        // finite candidate replace it while the final state
                        // still reports overflow if no finite path exists.
                        *current_tolerance = if candidate.is_finite() {
                            non_negative_score_tolerance(candidate)
                        } else {
                            0.0
                        };
                        unsafe {
                            *current_predecessors.get_unchecked_mut(current_changes) = start;
                        }
                        continue;
                    }
                    // Starts are visited in ascending order.  On an objective
                    // tie, retain the first predecessor instead of rebuilding
                    // and comparing complete paths in this hot loop.
                    if candidate < *current && *current - candidate > *current_tolerance {
                        *current = candidate;
                        *current_tolerance = non_negative_score_tolerance(candidate);
                        // SAFETY: predecessors has the same row width as scores.
                        unsafe {
                            *current_predecessors.get_unchecked_mut(current_changes) = start;
                        }
                    }
                }
            }
        }

        let final_score = scores[self.n_samples * width + changes];
        if !final_score.is_finite() {
            return Err(Error::NonFiniteObjective { value: final_score });
        }
        let breakpoints = backtrack_fixed(self.n_samples, changes, &predecessors, width)?;
        Segmentation::new(
            breakpoints,
            final_score,
            final_score,
            self.n_samples,
            self.grid.min_size,
        )
    }

    pub fn predict_penalty(&self, penalty: f64) -> Result<Segmentation, Error> {
        validate_penalty(penalty)?;
        if self.n_samples < self.grid.min_size {
            return Err(Error::SegmentTooShort {
                start: 0,
                end: self.n_samples,
                length: self.n_samples,
                minimum: self.grid.min_size,
            });
        }

        let mut best = vec![f64::INFINITY; self.n_samples + 1];
        let mut predecessors = vec![NO_PREDECESSOR; self.n_samples + 1];
        let mut diagonal_prefix = vec![0.0; self.n_samples + 1];
        let mut block_sums = vec![0.0; self.n_samples + 1];
        let mut active = vec![0usize];
        let mut prune_at = vec![usize::MAX; self.n_samples + 1];
        let mut evaluated = Vec::with_capacity(self.n_samples + 1);
        best[0] = -penalty;

        for end in 1..=self.n_samples {
            active.retain(|&start| prune_at[start] > end);
            let earliest_active = active.first().copied().unwrap_or(end - 1);
            self.extend_kernel_sums_from(
                end,
                earliest_active,
                &mut diagonal_prefix,
                &mut block_sums,
            );
            if !self.on_grid(end) {
                continue;
            }
            evaluated.clear();
            let mut end_best = f64::INFINITY;
            let mut end_predecessor = NO_PREDECESSOR;
            let end_diagonal = diagonal_prefix[end];

            for &start in &active {
                if end - start < self.grid.min_size {
                    continue;
                }
                let segment_cost = kernel_segment_cost(
                    end_diagonal - diagonal_prefix[start],
                    block_sums[start],
                    end - start,
                )?;
                let pruning_score = best[start] + segment_cost;
                evaluated.push((start, pruning_score));
                let candidate = pruning_score + penalty;
                if candidate < end_best
                    && (!end_best.is_finite()
                        || non_negative_candidate_is_significant(candidate, end_best))
                {
                    end_best = candidate;
                    end_predecessor = start;
                }
            }
            best[end] = end_best;
            predecessors[end] = end_predecessor;

            if end_best.is_finite() {
                for &(start, pruning_score) in &evaluated {
                    if pruning_score > end_best
                        && non_negative_increase_is_significant(pruning_score, end_best)
                    {
                        let deadline = &mut prune_at[start];
                        *deadline = (*deadline).min(end.saturating_add(self.grid.min_size));
                    }
                }
                active.push(end);
            }
        }

        if !best[self.n_samples].is_finite() {
            return Err(Error::NumericalFailure {
                context: "finding a feasible fused kernel segmentation",
            });
        }
        let breakpoints = backtrack_penalized(self.n_samples, &predecessors)?;
        let changes = breakpoints.len() - 1;
        let segment_cost = best[self.n_samples] - penalty * changes as f64;
        Segmentation::new(
            breakpoints,
            segment_cost,
            best[self.n_samples],
            self.n_samples,
            self.grid.min_size,
        )
    }

    fn extend_kernel_sums(&self, end: usize, diagonal_prefix: &mut [f64], block_sums: &mut [f64]) {
        self.extend_kernel_sums_from(end, 0, diagonal_prefix, block_sums);
    }

    fn extend_kernel_sums_from(
        &self,
        end: usize,
        minimum_start: usize,
        diagonal_prefix: &mut [f64],
        block_sums: &mut [f64],
    ) {
        let last = end - 1;
        debug_assert!(minimum_start <= last);
        let last_offset = last * self.n_features;
        let minimum_offset = minimum_start * self.n_features;
        let (_, from_minimum) = self.values.split_at(minimum_offset);
        let (previous_values, last_and_later_values) =
            from_minimum.split_at(last_offset - minimum_offset);
        let last_row = &last_and_later_values[..self.n_features];
        let diagonal = self.kernel.diagonal(last_row);
        diagonal_prefix[end] = diagonal_prefix[last] + diagonal;
        let mut cross_sum = diagonal;
        block_sums[last] += diagonal;
        for (block_sum, row) in block_sums[minimum_start..last]
            .iter_mut()
            .rev()
            .zip(previous_values.chunks_exact(self.n_features).rev())
        {
            cross_sum += self.kernel.similarity(row, last_row);
            *block_sum += 2.0 * cross_sum - diagonal;
        }
    }

    fn on_grid(&self, position: usize) -> bool {
        position == self.n_samples || position % self.grid.jump == 0
    }

    fn validate_feasible(&self, changes: usize) -> Result<usize, Error> {
        if self.n_samples < self.grid.min_size {
            return Err(self.infeasible(changes));
        }
        let spacing = self
            .grid
            .min_size
            .div_ceil(self.grid.jump)
            .checked_mul(self.grid.jump)
            .ok_or_else(|| self.infeasible(changes))?;
        let required = changes
            .checked_mul(spacing)
            .and_then(|value| value.checked_add(self.grid.min_size))
            .ok_or_else(|| self.infeasible(changes))?;
        if required > self.n_samples {
            return Err(self.infeasible(changes));
        }
        Ok(spacing)
    }

    fn infeasible(&self, changes: usize) -> Error {
        Error::InfeasibleSegmentation {
            n_samples: self.n_samples,
            changes,
            min_size: self.grid.min_size,
            jump: self.grid.jump,
        }
    }
}

fn kernel_segment_cost(diagonal: f64, block_sum: f64, length: usize) -> Result<f64, Error> {
    let value = diagonal - block_sum / length as f64;
    if !value.is_finite() {
        Err(Error::NonFiniteObjective { value })
    } else if value < -1.0e-10 {
        Err(Error::NumericalFailure {
            context: "computing a non-negative kernel segment cost",
        })
    } else if value < 0.0 {
        Ok(0.0)
    } else {
        Ok(value)
    }
}

fn build_fixed_path(
    mut end: usize,
    mut changes: usize,
    predecessors: &[usize],
    width: usize,
    output: &mut Vec<usize>,
) -> Result<(), Error> {
    output.clear();
    loop {
        output.push(end);
        if changes == 0 {
            break;
        }
        end = predecessors[end * width + changes];
        if end == NO_PREDECESSOR {
            return Err(Error::NumericalFailure {
                context: "reconstructing a fused fixed-K path",
            });
        }
        changes -= 1;
    }
    output.reverse();
    Ok(())
}

fn backtrack_fixed(
    end: usize,
    changes: usize,
    predecessors: &[usize],
    width: usize,
) -> Result<Vec<usize>, Error> {
    let mut output = Vec::with_capacity(changes + 1);
    build_fixed_path(end, changes, predecessors, width, &mut output)?;
    Ok(output)
}

fn build_penalized_path(
    mut end: usize,
    predecessors: &[usize],
    output: &mut Vec<usize>,
) -> Result<(), Error> {
    output.clear();
    while end != 0 {
        output.push(end);
        end = predecessors[end];
        if end == NO_PREDECESSOR {
            return Err(Error::NumericalFailure {
                context: "reconstructing a fused penalized path",
            });
        }
    }
    output.reverse();
    Ok(())
}

fn backtrack_penalized(end: usize, predecessors: &[usize]) -> Result<Vec<usize>, Error> {
    let mut output = Vec::new();
    build_penalized_path(end, predecessors, &mut output)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CosineKernel, Dynp, FullGramPrefix, LinearKernel, Pelt, RbfKernel};

    #[test]
    fn fused_matches_full_backend_for_all_kernels() {
        let values = [0., 0., 0., 4., 4., 4., -3., -3., -3.];
        let signal = SignalView::new(&values, 9, 1).unwrap();
        macro_rules! compare {
            ($kernel:expr) => {{
                let full = FullGramPrefix::fit(signal, $kernel, usize::MAX).unwrap();
                let fused = FusedKernelCPD::fit(signal, $kernel, 1, 1).unwrap();
                let expected_fixed = Dynp::new(1, 1).unwrap().predict_changes(&full, 2).unwrap();
                let actual_fixed = fused.predict_changes(2).unwrap();
                assert_eq!(actual_fixed.breakpoints, expected_fixed.breakpoints);
                assert!((actual_fixed.objective - expected_fixed.objective).abs() < 1e-10);
                let expected_penalty = Pelt::new(1, 1)
                    .unwrap()
                    .predict_penalty(&full, 1.0)
                    .unwrap();
                let actual_penalty = fused.predict_penalty(1.0).unwrap();
                assert_eq!(actual_penalty.breakpoints, expected_penalty.breakpoints);
                assert!((actual_penalty.objective - expected_penalty.objective).abs() < 1e-10);
            }};
        }
        compare!(LinearKernel);
        compare!(RbfKernel::new(0.5).unwrap());
        compare!(CosineKernel);
    }

    #[test]
    fn fused_grid_and_first_predecessor_ties_match_generic_dynp() {
        let values = [4.0; 8];
        let signal = SignalView::new(&values, 8, 1).unwrap();
        let full = FullGramPrefix::fit(signal, LinearKernel, usize::MAX).unwrap();
        let fused = FusedKernelCPD::fit(signal, LinearKernel, 1, 2).unwrap();
        for changes in 0..=3 {
            let expected = Dynp::new(1, 2)
                .unwrap()
                .predict_changes(&full, changes)
                .unwrap();
            let actual = fused.predict_changes(changes).unwrap();
            assert_eq!(actual.breakpoints, expected.breakpoints);
        }
    }

    #[test]
    fn fused_matches_generic_solvers_on_small_rbf_problems() {
        for n_samples in 2..=10 {
            let values: Vec<f64> = (0..n_samples)
                .map(|index| ((index * 13 + n_samples * 5) % 17) as f64)
                .collect();
            let signal = SignalView::new(&values, n_samples, 1).unwrap();
            let kernel = RbfKernel::new(0.3).unwrap();
            let full = FullGramPrefix::fit(signal, kernel, usize::MAX).unwrap();
            for min_size in 1..=n_samples.min(3) {
                for jump in 1..=3 {
                    let fused = FusedKernelCPD::fit(signal, kernel, min_size, jump).unwrap();
                    let maximum_changes = n_samples / min_size - 1;
                    for changes in 0..=maximum_changes {
                        let expected = Dynp::new(min_size, jump)
                            .unwrap()
                            .predict_changes(&full, changes);
                        let actual = fused.predict_changes(changes);
                        match (actual, expected) {
                            (Ok(actual), Ok(expected)) => {
                                assert_eq!(actual.breakpoints, expected.breakpoints);
                                assert!(
                                    (actual.objective - expected.objective).abs() < 1e-9
                                );
                            }
                            (Err(Error::InfeasibleSegmentation { .. }), Err(Error::InfeasibleSegmentation { .. })) => {}
                            (actual, expected) => panic!("fused={actual:?}, generic={expected:?}, n={n_samples}, min_size={min_size}, jump={jump}, changes={changes}"),
                        }
                    }
                    for penalty in [0.1, 1.0, 5.0] {
                        let expected = Pelt::new(min_size, jump)
                            .unwrap()
                            .predict_penalty(&full, penalty)
                            .unwrap();
                        let actual = fused.predict_penalty(penalty).unwrap();
                        assert_eq!(actual.breakpoints, expected.breakpoints);
                        assert!((actual.objective - expected.objective).abs() < 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn finite_inputs_with_non_finite_linear_arithmetic_are_rejected() {
        let values = [f64::MAX, -f64::MAX];
        let signal = SignalView::new(&values, 2, 1).unwrap();
        let linear = FusedKernelCPD::fit(signal, LinearKernel, 1, 1).unwrap();
        assert!(matches!(
            linear.predict_changes(0),
            Err(Error::NonFiniteObjective { .. })
        ));

        let rbf = FusedKernelCPD::fit(signal, RbfKernel::new(0.5).unwrap(), 1, 1).unwrap();
        assert!(rbf.predict_changes(0).is_ok());
    }
}
