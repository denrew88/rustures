use crate::core::segmentation::{
    non_negative_candidate_is_significant, non_negative_increase_is_significant,
    non_negative_score_tolerance,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::core::segmentation::{SCORE_ABSOLUTE_TOLERANCE, SCORE_RELATIVE_TOLERANCE};
use crate::{validate_penalty, Error, Kernel, SearchGrid, Segmentation, SignalView};

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    _mm256_add_pd, _mm256_and_pd, _mm256_blendv_pd, _mm256_cmp_pd, _mm256_loadu_pd,
    _mm256_movemask_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd, _CMP_GT_OQ,
    _CMP_LT_OQ,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_add_pd, _mm256_and_pd, _mm256_blendv_pd, _mm256_cmp_pd, _mm256_loadu_pd,
    _mm256_movemask_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd, _CMP_GT_OQ,
    _CMP_LT_OQ,
};

const NO_PREDECESSOR: usize = usize::MAX;
#[cfg(any(test, target_arch = "x86", target_arch = "x86_64"))]
const AVX2_F64_LANES: usize = 4;
#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
const AVX2_START_TILE: usize = 16;

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

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if changes >= AVX2_F64_LANES && std::is_x86_feature_detected!("avx2") {
            // SAFETY: the runtime feature check above proves that every AVX2
            // instruction used by this complete solver is supported. Dispatch
            // happens once per prediction, outside the end/start hot loops.
            return unsafe {
                self.predict_changes_avx2_start_major::<true>(changes, segment_count, spacing)
            };
        }

        self.predict_changes_scalar(changes, segment_count, spacing)
    }

    fn predict_changes_scalar(
        &self,
        changes: usize,
        width: usize,
        spacing: usize,
    ) -> Result<Segmentation, Error> {
        // Store states by the number of breakpoints, as `0..=changes`.
        // The previous representation stored segment counts and needed an
        // unused extra column, making every row `changes + 2` wide.
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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn predict_changes_avx2_start_major<const OPTIMIZED_UPDATES: bool>(
        &self,
        changes: usize,
        width: usize,
        spacing: usize,
    ) -> Result<Segmentation, Error> {
        debug_assert!(changes >= AVX2_F64_LANES);
        debug_assert_eq!(width, changes + 1);

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

        // AVX2 does not imply FMA and these operations intentionally mirror
        // the scalar `absolute + relative * score` tolerance expression.
        let absolute_tolerance = _mm256_set1_pd(SCORE_ABSOLUTE_TOLERANCE);
        let relative_tolerance = _mm256_set1_pd(SCORE_RELATIVE_TOLERANCE);

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

            current_scores[0] = kernel_segment_cost(end_diagonal, block_sums[0], end)?;
            if latest_start < spacing {
                continue;
            }

            for start in (spacing..=latest_start).step_by(self.grid.jump) {
                debug_assert!(start < end);
                debug_assert!(start < diagonal_prefix.len());
                debug_assert!(start < block_sums.len());
                // SAFETY: `start <= latest_start < end <= n_samples`, and all
                // accumulation arrays have `n_samples + 1` entries.
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
                // SAFETY: `start < end`, so the complete previous row is
                // disjoint from and lies before the mutable current row.
                let previous_scores = unsafe {
                    previous_score_rows.get_unchecked(start_offset..start_offset + width)
                };
                let feasible_changes = start / spacing;
                let maximum_changes = changes.min(feasible_changes);
                let initializes_new_state =
                    start == feasible_changes * spacing && feasible_changes <= changes;
                let regular_maximum = if initializes_new_state {
                    feasible_changes - 1
                } else {
                    maximum_changes
                };
                debug_assert!(maximum_changes < width);

                let segment_cost_vector = _mm256_set1_pd(segment_cost);
                let mut current_changes = 1usize;
                while current_changes + (AVX2_F64_LANES - 1) <= regular_maximum {
                    debug_assert!(current_changes + AVX2_F64_LANES <= width);
                    // SAFETY: the loop condition proves that four current,
                    // tolerance, and predecessor states fit in `0..width`.
                    // The preceding states start one element earlier and also
                    // fit in the complete immutable previous row. Unaligned
                    // loads are required because state 1 need not be aligned.
                    let (candidate, current, tolerance, update_mask, update_mask_bits) = unsafe {
                        let previous =
                            _mm256_loadu_pd(previous_scores.as_ptr().add(current_changes - 1));
                        let candidate = _mm256_add_pd(previous, segment_cost_vector);
                        let current = _mm256_loadu_pd(current_scores.as_ptr().add(current_changes));
                        let tolerance =
                            _mm256_loadu_pd(current_tolerances.as_ptr().add(current_changes));
                        let gap = _mm256_sub_pd(current, candidate);
                        // Tolerances are non-negative, so a significant
                        // positive gap already proves `candidate < current`.
                        // The ordered comparison also rejects the only NaN
                        // case here, `+infinity - +infinity`.
                        let gap_is_significant = _mm256_cmp_pd(gap, tolerance, _CMP_GT_OQ);
                        let update_mask = if OPTIMIZED_UPDATES {
                            gap_is_significant
                        } else {
                            let candidate_is_lower = _mm256_cmp_pd(candidate, current, _CMP_LT_OQ);
                            _mm256_and_pd(candidate_is_lower, gap_is_significant)
                        };
                        let update_mask_bits = _mm256_movemask_pd(update_mask) as u32;
                        (candidate, current, tolerance, update_mask, update_mask_bits)
                    };

                    // Most later candidates do not improve any of the four
                    // states. Avoid score/tolerance writes and tolerance
                    // arithmetic for those all-false masks. The all-true case
                    // can also store candidates directly without a blend.
                    if update_mask_bits != 0 {
                        unsafe {
                            let selected_scores = if update_mask_bits == 0b1111 {
                                candidate
                            } else {
                                _mm256_blendv_pd(current, candidate, update_mask)
                            };
                            _mm256_storeu_pd(
                                current_scores.as_mut_ptr().add(current_changes),
                                selected_scores,
                            );

                            let candidate_tolerance = _mm256_add_pd(
                                _mm256_mul_pd(candidate, relative_tolerance),
                                absolute_tolerance,
                            );
                            let selected_tolerances = if update_mask_bits == 0b1111 {
                                candidate_tolerance
                            } else {
                                _mm256_blendv_pd(tolerance, candidate_tolerance, update_mask)
                            };
                            _mm256_storeu_pd(
                                current_tolerances.as_mut_ptr().add(current_changes),
                                selected_tolerances,
                            );
                        }
                    }

                    // LLVM turns the common all-lanes case into one broadcast
                    // plus one vector store. Sparse parent writes keep the
                    // scalar bit scan to avoid touching unchanged lanes.
                    if OPTIMIZED_UPDATES && update_mask_bits == 0b1111 {
                        unsafe {
                            let parents = current_predecessors.as_mut_ptr().add(current_changes);
                            parents.write(start);
                            parents.add(1).write(start);
                            parents.add(2).write(start);
                            parents.add(3).write(start);
                        }
                    } else {
                        let mut changed_lanes = update_mask_bits;
                        while changed_lanes != 0 {
                            let lane = changed_lanes.trailing_zeros() as usize;
                            // SAFETY: movemask has exactly four meaningful
                            // bits, and the vector-range assertion covers it.
                            unsafe {
                                *current_predecessors.get_unchecked_mut(current_changes + lane) =
                                    start;
                            }
                            changed_lanes &= changed_lanes - 1;
                        }
                    }
                    current_changes += AVX2_F64_LANES;
                }

                // Handle up to three regular states after the packed blocks.
                while current_changes <= regular_maximum {
                    // SAFETY: `current_changes <= maximum_changes < width` and
                    // the previous state is in the complete previous row.
                    let previous = unsafe { *previous_scores.get_unchecked(current_changes - 1) };
                    let candidate = previous + segment_cost;
                    let current = unsafe { current_scores.get_unchecked_mut(current_changes) };
                    let current_tolerance =
                        unsafe { current_tolerances.get_unchecked_mut(current_changes) };
                    if candidate < *current && *current - candidate > *current_tolerance {
                        *current = candidate;
                        *current_tolerance = non_negative_score_tolerance(candidate);
                        unsafe {
                            *current_predecessors.get_unchecked_mut(current_changes) = start;
                        }
                    }
                    current_changes += 1;
                }

                // The first feasible candidate for a newly reachable state is
                // initialized directly. Keeping it outside vector blocks
                // preserves the scalar infinity/overflow contract exactly.
                if initializes_new_state {
                    let state = feasible_changes;
                    // SAFETY: `state <= changes < width`, and `state >= 1` for
                    // every start in this loop.
                    let previous = unsafe { *previous_scores.get_unchecked(state - 1) };
                    let candidate = previous + segment_cost;
                    unsafe {
                        *current_scores.get_unchecked_mut(state) = candidate;
                        *current_tolerances.get_unchecked_mut(state) = if candidate.is_finite() {
                            non_negative_score_tolerance(candidate)
                        } else {
                            0.0
                        };
                        *current_predecessors.get_unchecked_mut(state) = start;
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

    #[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
    #[target_feature(enable = "avx2")]
    unsafe fn predict_changes_avx2_tiled(
        &self,
        changes: usize,
        width: usize,
        spacing: usize,
    ) -> Result<Segmentation, Error> {
        debug_assert!(changes >= AVX2_F64_LANES);
        debug_assert_eq!(width, changes + 1);

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
        let mut tile_starts = [0usize; AVX2_START_TILE];
        let mut tile_costs = [0.0; AVX2_START_TILE];

        let absolute_tolerance = _mm256_set1_pd(SCORE_ABSOLUTE_TOLERANCE);
        let relative_tolerance = _mm256_set1_pd(SCORE_RELATIVE_TOLERANCE);

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

            current_scores[0] = kernel_segment_cost(end_diagonal, block_sums[0], end)?;
            if latest_start < spacing {
                continue;
            }

            let mut start_iter = (spacing..=latest_start).step_by(self.grid.jump);
            loop {
                let mut tile_len = 0usize;
                while tile_len < AVX2_START_TILE {
                    let Some(start) = start_iter.next() else {
                        break;
                    };
                    // SAFETY: the iterator is bounded by `latest_start < end`,
                    // and both accumulation arrays have `n_samples + 1` entries.
                    let (diagonal_start, block_sum) = unsafe {
                        (
                            *diagonal_prefix.get_unchecked(start),
                            *block_sums.get_unchecked(start),
                        )
                    };
                    tile_starts[tile_len] = start;
                    tile_costs[tile_len] =
                        kernel_segment_cost(end_diagonal - diagonal_start, block_sum, end - start)?;
                    tile_len += 1;
                }
                if tile_len == 0 {
                    break;
                }

                // Feasible state counts are monotone in ascending start order.
                // Vectorize the state prefix that every start in this tile can
                // update, keeping its best values live for the complete tile.
                let first_start = tile_starts[0];
                let first_feasible_changes = first_start / spacing;
                let first_maximum_changes = changes.min(first_feasible_changes);
                let first_initializes_new_state = first_start == first_feasible_changes * spacing
                    && first_feasible_changes <= changes;
                let common_regular_maximum = if first_initializes_new_state {
                    first_feasible_changes - 1
                } else {
                    first_maximum_changes
                };

                let mut current_changes = 1usize;
                while current_changes + (AVX2_F64_LANES - 1) <= common_regular_maximum {
                    debug_assert!(current_changes + AVX2_F64_LANES <= width);
                    // SAFETY: the vector loop condition keeps all four lanes
                    // within the current score, tolerance, and parent rows.
                    let (mut best_scores, mut best_tolerances) = unsafe {
                        (
                            _mm256_loadu_pd(current_scores.as_ptr().add(current_changes)),
                            _mm256_loadu_pd(current_tolerances.as_ptr().add(current_changes)),
                        )
                    };
                    let mut best_predecessors = unsafe {
                        [
                            *current_predecessors.get_unchecked(current_changes),
                            *current_predecessors.get_unchecked(current_changes + 1),
                            *current_predecessors.get_unchecked(current_changes + 2),
                            *current_predecessors.get_unchecked(current_changes + 3),
                        ]
                    };

                    for tile_index in 0..tile_len {
                        let start = tile_starts[tile_index];
                        let start_offset = start * width;
                        debug_assert!(start_offset + width <= previous_score_rows.len());
                        // SAFETY: every tile start is below `end`, and the
                        // four preceding states fit in its complete score row.
                        let candidate = unsafe {
                            let previous = _mm256_loadu_pd(
                                previous_score_rows
                                    .as_ptr()
                                    .add(start_offset + current_changes - 1),
                            );
                            _mm256_add_pd(previous, _mm256_set1_pd(tile_costs[tile_index]))
                        };
                        let gap = _mm256_sub_pd(best_scores, candidate);
                        let update_mask = _mm256_cmp_pd(gap, best_tolerances, _CMP_GT_OQ);
                        let update_mask_bits = _mm256_movemask_pd(update_mask) as u32;
                        if update_mask_bits == 0 {
                            continue;
                        }

                        best_scores = if update_mask_bits == 0b1111 {
                            candidate
                        } else {
                            _mm256_blendv_pd(best_scores, candidate, update_mask)
                        };
                        let candidate_tolerance = _mm256_add_pd(
                            _mm256_mul_pd(candidate, relative_tolerance),
                            absolute_tolerance,
                        );
                        best_tolerances = if update_mask_bits == 0b1111 {
                            candidate_tolerance
                        } else {
                            _mm256_blendv_pd(best_tolerances, candidate_tolerance, update_mask)
                        };

                        let mut changed_lanes = update_mask_bits;
                        while changed_lanes != 0 {
                            let lane = changed_lanes.trailing_zeros() as usize;
                            best_predecessors[lane] = start;
                            changed_lanes &= changed_lanes - 1;
                        }
                    }

                    // SAFETY: the same four-lane range was established before
                    // the tile scan. Each array has the complete state width.
                    unsafe {
                        _mm256_storeu_pd(
                            current_scores.as_mut_ptr().add(current_changes),
                            best_scores,
                        );
                        _mm256_storeu_pd(
                            current_tolerances.as_mut_ptr().add(current_changes),
                            best_tolerances,
                        );
                        let parents = current_predecessors.as_mut_ptr().add(current_changes);
                        parents.write(best_predecessors[0]);
                        parents.add(1).write(best_predecessors[1]);
                        parents.add(2).write(best_predecessors[2]);
                        parents.add(3).write(best_predecessors[3]);
                    }
                    current_changes += AVX2_F64_LANES;
                }

                // The non-common diagonal and at most three state-tail lanes
                // remain scalar. Starts stay in ascending order, preserving
                // the exact first-predecessor and tolerance semantics.
                for tile_index in 0..tile_len {
                    let start = tile_starts[tile_index];
                    let segment_cost = tile_costs[tile_index];
                    let start_offset = start * width;
                    let previous_scores = unsafe {
                        previous_score_rows.get_unchecked(start_offset..start_offset + width)
                    };
                    let feasible_changes = start / spacing;
                    let maximum_changes = changes.min(feasible_changes);
                    let initializes_new_state =
                        start == feasible_changes * spacing && feasible_changes <= changes;
                    let regular_maximum = if initializes_new_state {
                        feasible_changes - 1
                    } else {
                        maximum_changes
                    };
                    let mut scalar_changes = current_changes;
                    while scalar_changes <= regular_maximum {
                        let previous =
                            unsafe { *previous_scores.get_unchecked(scalar_changes - 1) };
                        let candidate = previous + segment_cost;
                        let current = unsafe { current_scores.get_unchecked_mut(scalar_changes) };
                        let current_tolerance =
                            unsafe { current_tolerances.get_unchecked_mut(scalar_changes) };
                        if candidate < *current && *current - candidate > *current_tolerance {
                            *current = candidate;
                            *current_tolerance = non_negative_score_tolerance(candidate);
                            unsafe {
                                *current_predecessors.get_unchecked_mut(scalar_changes) = start;
                            }
                        }
                        scalar_changes += 1;
                    }

                    if initializes_new_state {
                        let state = feasible_changes;
                        let previous = unsafe { *previous_scores.get_unchecked(state - 1) };
                        let candidate = previous + segment_cost;
                        unsafe {
                            *current_scores.get_unchecked_mut(state) = candidate;
                            *current_tolerances.get_unchecked_mut(state) = if candidate.is_finite()
                            {
                                non_negative_score_tolerance(candidate)
                            } else {
                                0.0
                            };
                            *current_predecessors.get_unchecked_mut(state) = start;
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
    use crate::{
        CosineKernel, Dynp, FullGramPrefix, FusedKernel, GammaPolicy, KernelKind, LinearKernel,
        Pelt, RbfKernel,
    };

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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_fixed_k_matches_scalar_for_all_kernels() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let values: Vec<f64> = (0..65)
            .flat_map(|row| {
                let row = row as f64;
                [
                    (row * 0.7).sin() + 0.25,
                    (row * 1.3).cos() - 0.5,
                    (row % 4.0) - 1.25,
                ]
            })
            .collect();
        let signal = SignalView::new(&values, 65, 3).unwrap();

        fn compare<K: Kernel>(signal: SignalView<'_>, kernel: K) {
            for min_size in 1..=2 {
                for jump in 1..=2 {
                    let fused =
                        FusedKernelCPD::fit(signal, kernel.clone(), min_size, jump).unwrap();
                    let maximum_changes = signal.shape().n_samples / min_size - 1;
                    for changes in [4usize, 5, 8, 16, 32, 64]
                        .into_iter()
                        .filter(|&changes| changes <= maximum_changes)
                    {
                        let width = changes + 1;
                        let spacing = match fused.validate_feasible(changes) {
                            Ok(spacing) => spacing,
                            Err(Error::InfeasibleSegmentation { .. }) => continue,
                            Err(error) => panic!("unexpected feasibility error: {error}"),
                        };
                        let scalar = fused
                            .predict_changes_scalar(changes, width, spacing)
                            .unwrap();
                        // SAFETY: the test returns early on CPUs without AVX2.
                        let start_major = unsafe {
                            fused.predict_changes_avx2_start_major::<true>(changes, width, spacing)
                        }
                        .unwrap();
                        let baseline = unsafe {
                            fused.predict_changes_avx2_start_major::<false>(changes, width, spacing)
                        }
                        .unwrap();
                        let tiled =
                            unsafe { fused.predict_changes_avx2_tiled(changes, width, spacing) }
                                .unwrap();
                        assert_eq!(start_major.breakpoints, scalar.breakpoints);
                        assert_eq!(start_major.objective.to_bits(), scalar.objective.to_bits());
                        assert_eq!(baseline.breakpoints, scalar.breakpoints);
                        assert_eq!(baseline.objective.to_bits(), scalar.objective.to_bits());
                        assert_eq!(tiled.breakpoints, scalar.breakpoints);
                        assert_eq!(tiled.objective.to_bits(), scalar.objective.to_bits());
                    }
                }
            }
        }

        compare(signal, LinearKernel);
        compare(signal, RbfKernel::new(0.35).unwrap());
        compare(signal, CosineKernel);
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

    #[derive(Clone, Copy, Default)]
    struct FixedKOperationCounts {
        segment_costs: usize,
        vector_blocks: usize,
        vector_lanes: usize,
        scalar_tail_states: usize,
        direct_initializations: usize,
    }

    fn median_elapsed<T>(repeats: usize, mut operation: impl FnMut() -> T) -> std::time::Duration {
        let warmup = operation();
        std::hint::black_box(&warmup);
        drop(warmup);
        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let started = std::time::Instant::now();
            let output = operation();
            std::hint::black_box(&output);
            samples.push(started.elapsed());
            drop(output);
        }
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn median_three_elapsed<T>(
        repeats: usize,
        mut start_major: impl FnMut() -> T,
        mut tiled: impl FnMut() -> T,
        mut scalar: impl FnMut() -> T,
    ) -> (
        std::time::Duration,
        std::time::Duration,
        std::time::Duration,
    ) {
        fn measure<T>(operation: &mut impl FnMut() -> T) -> std::time::Duration {
            let started = std::time::Instant::now();
            let output = operation();
            std::hint::black_box(&output);
            let elapsed = started.elapsed();
            drop(output);
            elapsed
        }

        std::hint::black_box(start_major());
        std::hint::black_box(tiled());
        std::hint::black_box(scalar());
        let mut start_major_samples = Vec::with_capacity(repeats);
        let mut tiled_samples = Vec::with_capacity(repeats);
        let mut scalar_samples = Vec::with_capacity(repeats);
        for iteration in 0..repeats {
            match iteration % 3 {
                0 => {
                    start_major_samples.push(measure(&mut start_major));
                    tiled_samples.push(measure(&mut tiled));
                    scalar_samples.push(measure(&mut scalar));
                }
                1 => {
                    tiled_samples.push(measure(&mut tiled));
                    scalar_samples.push(measure(&mut scalar));
                    start_major_samples.push(measure(&mut start_major));
                }
                _ => {
                    scalar_samples.push(measure(&mut scalar));
                    start_major_samples.push(measure(&mut start_major));
                    tiled_samples.push(measure(&mut tiled));
                }
            }
        }
        start_major_samples.sort_unstable();
        tiled_samples.sort_unstable();
        scalar_samples.sort_unstable();
        (
            start_major_samples[start_major_samples.len() / 2],
            tiled_samples[tiled_samples.len() / 2],
            scalar_samples[scalar_samples.len() / 2],
        )
    }

    fn median_reported_duration(
        repeats: usize,
        mut operation: impl FnMut() -> std::time::Duration,
    ) -> std::time::Duration {
        std::hint::black_box(operation());
        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            samples.push(operation());
        }
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    fn median_allocation_and_drop(
        n_samples: usize,
        width: usize,
        repeats: usize,
    ) -> (std::time::Duration, std::time::Duration) {
        let table_len = (n_samples + 1) * width;
        let mut allocation_samples = Vec::with_capacity(repeats);
        let mut drop_samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let started = std::time::Instant::now();
            let scores = vec![f64::INFINITY; table_len];
            let predecessors = vec![NO_PREDECESSOR; table_len];
            let tolerances = vec![0.0; width];
            let diagonal = vec![0.0; n_samples + 1];
            let block_sums = vec![0.0; n_samples + 1];
            std::hint::black_box((
                scores.as_ptr(),
                predecessors.as_ptr(),
                tolerances.as_ptr(),
                diagonal.as_ptr(),
                block_sums.as_ptr(),
            ));
            allocation_samples.push(started.elapsed());

            let started = std::time::Instant::now();
            drop((scores, predecessors, tolerances, diagonal, block_sums));
            drop_samples.push(started.elapsed());
        }
        allocation_samples.sort_unstable();
        drop_samples.sort_unstable();
        (allocation_samples[repeats / 2], drop_samples[repeats / 2])
    }

    fn measure_kernel_accumulation<K: Kernel>(
        detector: &FusedKernelCPD<K>,
        repeats: usize,
    ) -> std::time::Duration {
        median_reported_duration(repeats, || {
            let mut diagonal = vec![0.0; detector.n_samples + 1];
            let mut block_sums = vec![0.0; detector.n_samples + 1];
            let started = std::time::Instant::now();
            for end in 1..=detector.n_samples {
                detector.extend_kernel_sums(end, &mut diagonal, &mut block_sums);
            }
            let elapsed = started.elapsed();
            std::hint::black_box((diagonal, block_sums));
            elapsed
        })
    }

    fn timer_loop_overhead(iterations: usize) -> std::time::Duration {
        let mut elapsed = std::time::Duration::ZERO;
        for _ in 0..iterations {
            let started = std::time::Instant::now();
            std::hint::black_box(());
            elapsed += started.elapsed();
        }
        elapsed
    }

    fn measure_segment_cost_scan<K: Kernel>(
        detector: &FusedKernelCPD<K>,
        spacing: usize,
        repeats: usize,
    ) -> std::time::Duration {
        median_reported_duration(repeats, || {
            let mut diagonal = vec![0.0; detector.n_samples + 1];
            let mut block_sums = vec![0.0; detector.n_samples + 1];
            let mut elapsed = std::time::Duration::ZERO;
            let mut timed_ends = 0usize;
            let mut checksum = 0.0;
            for end in 1..=detector.n_samples {
                detector.extend_kernel_sums(end, &mut diagonal, &mut block_sums);
                if !detector.on_grid(end) || end < detector.grid.min_size {
                    continue;
                }
                let latest_start = end - detector.grid.min_size;
                if latest_start < spacing {
                    continue;
                }
                let end_diagonal = diagonal[end];
                let started = std::time::Instant::now();
                for start in (spacing..=latest_start).step_by(detector.grid.jump) {
                    checksum += kernel_segment_cost(
                        end_diagonal - diagonal[start],
                        block_sums[start],
                        end - start,
                    )
                    .unwrap();
                }
                elapsed += started.elapsed();
                timed_ends += 1;
            }
            std::hint::black_box(checksum);
            elapsed.saturating_sub(timer_loop_overhead(timed_ends))
        })
    }

    fn measure_backtrack(n_samples: usize, changes: usize, repeats: usize) -> std::time::Duration {
        let width = changes + 1;
        let mut predecessors = vec![NO_PREDECESSOR; (n_samples + 1) * width];
        let mut end = n_samples;
        for state in (1..=changes).rev() {
            predecessors[end * width + state] = end - 1;
            end -= 1;
        }
        let batch = 2_000usize;
        median_reported_duration(repeats, || {
            let started = std::time::Instant::now();
            for _ in 0..batch {
                std::hint::black_box(
                    backtrack_fixed(n_samples, changes, &predecessors, width).unwrap(),
                );
            }
            started.elapsed() / batch as u32
        })
    }

    fn fixed_k_operation_counts<K: Kernel>(
        detector: &FusedKernelCPD<K>,
        changes: usize,
        spacing: usize,
    ) -> FixedKOperationCounts {
        let mut counts = FixedKOperationCounts::default();
        for end in 1..=detector.n_samples {
            if !detector.on_grid(end) || end < detector.grid.min_size {
                continue;
            }
            let latest_start = end - detector.grid.min_size;
            if latest_start < spacing {
                continue;
            }
            for start in (spacing..=latest_start).step_by(detector.grid.jump) {
                counts.segment_costs += 1;
                let feasible_changes = start / spacing;
                let maximum_changes = changes.min(feasible_changes);
                let initializes_new_state =
                    start == feasible_changes * spacing && feasible_changes <= changes;
                let regular_maximum = if initializes_new_state {
                    feasible_changes - 1
                } else {
                    maximum_changes
                };
                let vector_blocks = regular_maximum / AVX2_F64_LANES;
                counts.vector_blocks += vector_blocks;
                counts.vector_lanes += vector_blocks * AVX2_F64_LANES;
                counts.scalar_tail_states += regular_maximum % AVX2_F64_LANES;
                counts.direct_initializations += usize::from(initializes_new_state);
            }
        }
        counts
    }

    fn milliseconds(duration: std::time::Duration) -> f64 {
        duration.as_secs_f64() * 1_000.0
    }

    fn profile_detector<K: Kernel>(
        name: &str,
        detector: &FusedKernelCPD<K>,
        changes: usize,
        fit: std::time::Duration,
        repeats: usize,
        kernel: std::time::Duration,
        segment_costs: std::time::Duration,
    ) {
        let spacing = detector.validate_feasible(changes).unwrap();
        let width = changes + 1;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let use_avx2 = changes >= AVX2_F64_LANES && std::is_x86_feature_detected!("avx2");
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let use_avx2 = false;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let (total, baseline_total, scalar_total) = if use_avx2 {
            median_three_elapsed(
                repeats,
                || unsafe {
                    detector
                        .predict_changes_avx2_start_major::<true>(changes, width, spacing)
                        .unwrap()
                },
                || unsafe {
                    detector
                        .predict_changes_avx2_start_major::<false>(changes, width, spacing)
                        .unwrap()
                },
                || {
                    detector
                        .predict_changes_scalar(changes, width, spacing)
                        .unwrap()
                },
            )
        } else {
            let total = median_elapsed(repeats, || detector.predict_changes(changes).unwrap());
            (total, total, total)
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let (total, baseline_total, scalar_total) = {
            let total = median_elapsed(repeats, || detector.predict_changes(changes).unwrap());
            (total, total, total)
        };
        let (allocation, table_drop) =
            median_allocation_and_drop(detector.n_samples, width, repeats);
        let backtrack = measure_backtrack(detector.n_samples, changes, repeats);
        let known = allocation + table_drop + kernel + segment_costs + backtrack;
        let k_state_residual = total.saturating_sub(known);
        let baseline_k_state_residual = baseline_total.saturating_sub(known);
        let scalar_k_state_residual = scalar_total.saturating_sub(known);
        let counts = fixed_k_operation_counts(detector, changes, spacing);
        let selected_backend = if use_avx2 { "avx2" } else { "scalar" };

        println!(
            "PROFILE kernel={name} n={} d={} K={changes} selected_backend={selected_backend} fit_ms={:.6} alloc_ms={:.6} drop_ms={:.6} kernel_ms={:.6} segment_ms={:.6} optimized_k_state_ms={:.6} baseline_k_state_ms={:.6} scalar_k_state_ms={:.6} backtrack_ms={:.6} optimized_total_ms={:.6} baseline_total_ms={:.6} scalar_total_ms={:.6} segment_calls={} vector_blocks={} vector_lanes={} scalar_tail={} direct_init={}",
            detector.n_samples,
            detector.n_features,
            milliseconds(fit),
            milliseconds(allocation),
            milliseconds(table_drop),
            milliseconds(kernel),
            milliseconds(segment_costs),
            milliseconds(k_state_residual),
            milliseconds(baseline_k_state_residual),
            milliseconds(scalar_k_state_residual),
            milliseconds(backtrack),
            milliseconds(total),
            milliseconds(baseline_total),
            milliseconds(scalar_total),
            counts.segment_costs,
            counts.vector_blocks,
            counts.vector_lanes,
            counts.scalar_tail_states,
            counts.direct_initializations,
        );
    }

    fn profile_detector_series<K: Kernel>(
        name: &str,
        detector: &FusedKernelCPD<K>,
        fit: std::time::Duration,
        repeats: usize,
    ) {
        let spacing = detector.validate_feasible(1).unwrap();
        let kernel = measure_kernel_accumulation(detector, repeats);
        let segment_costs = measure_segment_cost_scan(detector, spacing, repeats);
        let change_counts = std::env::var("RUSTURES_PROFILE_K")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|part| part.trim().parse().ok())
                    .collect::<Vec<_>>()
            })
            .filter(|values| !values.is_empty())
            .unwrap_or_else(|| vec![1usize, 2, 4, 8, 16, 32, 64]);
        for changes in change_counts {
            profile_detector(name, detector, changes, fit, repeats, kernel, segment_costs);
        }
    }

    fn profiling_signal(n_samples: usize, n_features: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(n_samples * n_features);
        for index in 0..n_samples {
            let segment = (index * 4 / n_samples).min(3);
            for feature in 0..n_features {
                let levels = [0.0, 5.0, -3.0, 7.0];
                let level = levels[(segment + 4 - feature % 4) % 4];
                let multiplier = 48_271usize + feature * 2_143;
                let noise = (((index * multiplier + index * index * (31 + feature * 2)) % 104_729)
                    as f64
                    / 104_729.0
                    - 0.5)
                    * 0.1;
                values.push(level + noise);
            }
        }
        values
    }

    #[test]
    #[ignore = "manual Rust-only fixed-K stage profiler"]
    fn profile_fixed_k_stages() {
        let repeats = std::env::var("RUSTURES_PROFILE_REPEATS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(15usize);
        let sample_sizes = std::env::var("RUSTURES_PROFILE_N")
            .ok()
            .and_then(|value| value.parse().ok())
            .map_or_else(|| vec![800usize, 1_600], |value| vec![value]);

        for n_samples in sample_sizes {
            let values = profiling_signal(n_samples, 2);
            let signal = SignalView::new(&values, n_samples, 2).unwrap();
            for (name, kind) in [
                ("linear", KernelKind::Linear),
                ("rbf", KernelKind::Rbf(GammaPolicy::Fixed(0.5))),
                ("cosine", KernelKind::Cosine),
            ] {
                let fit = median_elapsed(repeats, || FusedKernel::fit(signal, kind, 1, 1).unwrap());
                let fitted = FusedKernel::fit(signal, kind, 1, 1).unwrap();
                match &fitted {
                    FusedKernel::Linear(detector) => {
                        profile_detector_series(name, detector, fit, repeats)
                    }
                    FusedKernel::Rbf(detector) => {
                        profile_detector_series(name, detector, fit, repeats)
                    }
                    FusedKernel::Cosine(detector) => {
                        profile_detector_series(name, detector, fit, repeats)
                    }
                }
            }
        }
    }
}
