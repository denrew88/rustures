use super::*;
use crate::{
    CosineKernel, Dynp, FullGramPrefix, FusedKernel, GammaPolicy, KernelKind, LinearKernel, Pelt,
    RbfKernel,
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
                let fused = FusedKernelCPD::fit(signal, kernel.clone(), min_size, jump).unwrap();
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
    let (allocation, table_drop) = median_allocation_and_drop(detector.n_samples, width, repeats);
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
                FusedKernel::Rbf(detector) => profile_detector_series(name, detector, fit, repeats),
                FusedKernel::Cosine(detector) => {
                    profile_detector_series(name, detector, fit, repeats)
                }
            }
        }
    }
}
