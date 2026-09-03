use super::*;
use crate::oracle::{best_fixed_changes, best_penalized};
use crate::CostL2;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[test]
fn full_and_streaming_costs_match_for_all_kernels() {
    let values = [0., 0., 1., 0., 1., 1., 0., 1.];
    let signal = SignalView::new(&values, 4, 2).unwrap();
    macro_rules! compare {
        ($kernel:expr) => {{
            let full = FullGramPrefix::fit(signal, $kernel, usize::MAX).unwrap();
            let streaming = StreamingKernelCost::fit(signal, $kernel);
            for start in 0..4 {
                for end in start + 1..=4 {
                    assert!(
                        (full.cost(start..end).unwrap() - streaming.cost(start..end).unwrap())
                            .abs()
                            < 1e-12
                    );
                }
            }
        }};
    }
    compare!(LinearKernel);
    compare!(CosineKernel);
    compare!(RbfKernel::new(0.7).unwrap());
}

#[test]
fn streaming_endpoint_batches_match_scalar_costs_across_resets() {
    let values = [0.0, 1.0, 2.0, -1.0, 3.0, 0.5, -2.0, 4.0, 1.0, -3.0];
    let signal = SignalView::new(&values, 5, 2).unwrap();

    macro_rules! compare {
        ($kernel:expr) => {{
            let cost = StreamingKernelCost::fit(signal, $kernel);
            let mut actual = Vec::new();
            for (starts, end) in [
                (vec![0, 1, 3], 4),
                (vec![0, 1], 2),
                (vec![2, 0, 4], 5),
                (vec![1, 3], 5),
            ] {
                cost.costs_ending_at(&starts, end, &mut actual).unwrap();
                let expected: Vec<f64> = starts
                    .iter()
                    .map(|&start| cost.cost(start..end).unwrap())
                    .collect();
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-10);
                }
            }
        }};
    }

    compare!(LinearKernel);
    compare!(CosineKernel);
    compare!(RbfKernel::new(0.7).unwrap());
}

#[derive(Clone)]
struct CountingLinearKernel(Arc<AtomicUsize>);

impl Kernel for CountingLinearKernel {
    fn similarity(&self, left: &[f64], right: &[f64]) -> f64 {
        self.0.fetch_add(1, Ordering::Relaxed);
        left.iter().zip(right).map(|(x, y)| x * y).sum()
    }
}

#[test]
fn streaming_endpoint_sweep_computes_each_symmetric_pair_once() {
    let n_samples = 6;
    let values: Vec<f64> = (0..n_samples).map(|value| value as f64).collect();
    let signal = SignalView::new(&values, n_samples, 1).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let cost = StreamingKernelCost::fit(signal, CountingLinearKernel(calls.clone()));
    assert_eq!(calls.load(Ordering::Relaxed), n_samples);

    let mut output = Vec::new();
    for end in 1..=n_samples {
        let starts: Vec<usize> = (0..end).collect();
        cost.costs_ending_at(&starts, end, &mut output).unwrap();
    }

    // Fit evaluates n diagonal entries. The endpoint sweep then evaluates
    // exactly one triangular half, including its n diagonal entries.
    assert_eq!(
        calls.load(Ordering::Relaxed),
        n_samples + n_samples * (n_samples + 1) / 2
    );

    let calls_before_reuse = calls.load(Ordering::Relaxed);
    cost.costs_ending_at(&[0, 2, 5], n_samples, &mut output)
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), calls_before_reuse);
}

#[test]
fn full_gram_computes_each_symmetric_pair_once() {
    let n_samples = 7;
    let values: Vec<f64> = (0..n_samples).map(|value| value as f64).collect();
    let signal = SignalView::new(&values, n_samples, 1).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let full =
        FullGramPrefix::fit(signal, CountingLinearKernel(calls.clone()), usize::MAX).unwrap();
    assert_eq!(
        calls.load(Ordering::Relaxed),
        n_samples * (n_samples + 1) / 2
    );

    let streaming = StreamingKernelCost::fit(signal, LinearKernel);
    for start in 0..n_samples {
        for end in start + 1..=n_samples {
            assert!(
                (full.cost(start..end).unwrap() - streaming.cost(start..end).unwrap()).abs()
                    < 1e-10
            );
        }
    }
}

#[test]
fn full_gram_endpoint_batches_match_scalar_costs() {
    let values = [0.0, 1.0, 2.0, -1.0, 3.0, 0.5, -2.0, 4.0, 1.0, -3.0];
    let signal = SignalView::new(&values, 5, 2).unwrap();

    macro_rules! compare {
        ($kernel:expr) => {{
            let cost = FullGramPrefix::fit(signal, $kernel, usize::MAX).unwrap();
            let mut actual = Vec::new();
            for (starts, end) in [(vec![0, 1, 3], 4), (vec![0, 1], 2), (vec![2, 0, 4], 5)] {
                cost.costs_ending_at(&starts, end, &mut actual).unwrap();
                let expected: Vec<f64> = starts
                    .iter()
                    .map(|&start| cost.cost(start..end).unwrap())
                    .collect();
                assert_eq!(actual, expected);
            }
        }};
    }

    compare!(LinearKernel);
    compare!(CosineKernel);
    compare!(RbfKernel::new(0.7).unwrap());
}

#[test]
fn kernel_cost_enum_forwards_streaming_endpoint_batches() {
    let values = [0.0, 1.0, 4.0, -2.0, 3.0, 5.0];
    let signal = SignalView::new(&values, 6, 1).unwrap();
    let cost = KernelCost::fit(
        signal,
        KernelKind::Rbf(GammaPolicy::Fixed(0.4)),
        KernelBackend::Streaming,
        0,
    )
    .unwrap();
    let starts = [0, 1, 3, 5];
    let mut actual = Vec::new();
    cost.costs_ending_at(&starts, 6, &mut actual).unwrap();
    for (&start, actual) in starts.iter().zip(actual) {
        let expected = cost.cost(start..6).unwrap();
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn pre_normalized_cosine_backends_preserve_costs_and_public_shape() {
    let values = [0.0, 0.0, 1.0, 0.0, 3.0, 4.0, 0.0, 0.0, -2.0, 1.0];
    let signal = SignalView::new(&values, 5, 2).unwrap();
    let direct = StreamingKernelCost::fit(signal, CosineKernel);

    for backend in [KernelBackend::FullGram, KernelBackend::Streaming] {
        let optimized = KernelCost::fit(signal, KernelKind::Cosine, backend, usize::MAX).unwrap();
        assert_eq!(optimized.n_features(), 2);
        for start in 0..5 {
            for end in start + 1..=5 {
                assert!(
                    (optimized.cost(start..end).unwrap() - direct.cost(start..end).unwrap()).abs()
                        < 1e-10
                );
            }
        }
    }
}

#[test]
fn cosine_zero_vector_and_memory_limit_are_explicit() {
    assert_eq!(CosineKernel.similarity(&[0., 0.], &[0., 0.]), 1.0);
    assert_eq!(CosineKernel.similarity(&[0., 0.], &[1., 0.]), 0.0);
    let signal = SignalView::new(&[0., 1., 2.], 3, 1).unwrap();
    assert!(matches!(
        FullGramPrefix::fit(signal, LinearKernel, 8),
        Err(Error::GramMemoryLimit { .. })
    ));
}

#[test]
fn sampled_gamma_is_deterministic() {
    let signal = SignalView::new(&[0., 1., 3., 8.], 4, 1).unwrap();
    let policy = GammaPolicy::SampledMedian { pairs: 20, seed: 9 };
    assert_eq!(
        resolve_gamma(signal, policy).unwrap(),
        resolve_gamma(signal, policy).unwrap()
    );
}

#[test]
fn full_and_streaming_exact_detectors_match() {
    let values = [0., 0., 0., 4., 4., 4., -3., -3., -3.];
    let signal = SignalView::new(&values, 9, 1).unwrap();
    let kind = KernelKind::Rbf(GammaPolicy::Fixed(0.5));
    let full = KernelCPD::fit(signal, kind, KernelBackend::FullGram, 1, 1, usize::MAX).unwrap();
    let streaming = KernelCPD::fit(signal, kind, KernelBackend::Streaming, 1, 1, 0).unwrap();
    let full_fixed = full.predict_changes(2).unwrap();
    let streaming_fixed = streaming.predict_changes(2).unwrap();
    let brute_fixed = best_fixed_changes(9, 1, 2, |segment| full.cost().cost(segment).unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(full_fixed.breakpoints, streaming_fixed.breakpoints);
    assert_eq!(full_fixed.breakpoints, brute_fixed.0);
    assert!((full_fixed.objective - streaming_fixed.objective).abs() < 1e-10);
    assert!((full_fixed.objective - brute_fixed.1).abs() < 1e-10);
    let full_penalty = full.predict_penalty(1.0).unwrap();
    let streaming_penalty = streaming.predict_penalty(1.0).unwrap();
    let brute_penalty =
        best_penalized(9, 1, 1.0, |segment| full.cost().cost(segment).unwrap()).unwrap();
    assert_eq!(full_penalty.breakpoints, streaming_penalty.breakpoints);
    assert_eq!(full_penalty.breakpoints, brute_penalty.0);
    assert!((full_penalty.objective - streaming_penalty.objective).abs() < 1e-10);
    assert!((full_penalty.objective - brute_penalty.1).abs() < 1e-10);
    assert_eq!(streaming.cost().stored_gram_entries(), 0);
}

#[test]
fn linear_detectors_are_stable_under_large_feature_translations() {
    let base = [
        0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 5.0, -2.0, 5.0, -1.0, 5.0, -2.0, -3.0, 4.0, -3.0, 5.0, -3.0,
        4.0,
    ];
    let mut translated = base;
    for row in translated.chunks_exact_mut(2) {
        row[0] += 1.0e12;
        row[1] -= 1.0e12;
    }
    let base_signal = SignalView::new(&base, 9, 2).unwrap();
    let translated_signal = SignalView::new(&translated, 9, 2).unwrap();

    let base_fused = FusedKernel::fit(base_signal, KernelKind::Linear, 1, 1).unwrap();
    let translated_fused = FusedKernel::fit(translated_signal, KernelKind::Linear, 1, 1).unwrap();
    assert_eq!(
        base_fused.predict_changes(2).unwrap().breakpoints,
        translated_fused.predict_changes(2).unwrap().breakpoints
    );
    assert_eq!(
        base_fused.predict_penalty(1.0).unwrap().breakpoints,
        translated_fused.predict_penalty(1.0).unwrap().breakpoints
    );

    let l2 = CostL2::fit(translated_signal).unwrap();
    for backend in [KernelBackend::FullGram, KernelBackend::Streaming] {
        let base_detector =
            KernelCPD::fit(base_signal, KernelKind::Linear, backend, 1, 1, usize::MAX).unwrap();
        let translated_detector = KernelCPD::fit(
            translated_signal,
            KernelKind::Linear,
            backend,
            1,
            1,
            usize::MAX,
        )
        .unwrap();
        assert_eq!(
            base_detector.predict_changes(2).unwrap().breakpoints,
            translated_detector.predict_changes(2).unwrap().breakpoints
        );
        assert_eq!(
            base_detector.predict_penalty(1.0).unwrap().breakpoints,
            translated_detector
                .predict_penalty(1.0)
                .unwrap()
                .breakpoints
        );
        for start in 0..9 {
            for end in start + 1..=9 {
                let kernel_cost = translated_detector.cost().cost(start..end).unwrap();
                let l2_cost = l2.cost(start..end).unwrap();
                assert!(
                    (kernel_cost - l2_cost).abs() <= 1.0e-12,
                    "segment {start}..{end}: kernel={kernel_cost}, l2={l2_cost}"
                );
            }
        }
    }
}

#[test]
fn linear_centering_overflow_is_a_typed_error() {
    let values = [f64::MAX, -f64::MAX];
    let signal = SignalView::new(&values, 2, 1).unwrap();
    assert!(matches!(
        FusedKernel::fit(signal, KernelKind::Linear, 1, 1),
        Err(Error::NumericalFailure {
            context: "centering linear kernel observations"
        })
    ));
    assert!(matches!(
        KernelCost::fit(
            signal,
            KernelKind::Linear,
            KernelBackend::Streaming,
            usize::MAX
        ),
        Err(Error::NumericalFailure {
            context: "centering linear kernel observations"
        })
    ));
}
