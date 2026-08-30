use std::ops::Range;

use crate::{
    validate_segment, Dynp, Error, FusedKernelCPD, Pelt, SegmentCost, Segmentation, SignalView,
};

pub trait Kernel: Clone + Send + Sync {
    fn similarity(&self, left: &[f64], right: &[f64]) -> f64;

    fn diagonal(&self, value: &[f64]) -> f64 {
        self.similarity(value, value)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LinearKernel;

impl Kernel for LinearKernel {
    fn similarity(&self, left: &[f64], right: &[f64]) -> f64 {
        left.iter().zip(right).map(|(x, y)| x * y).sum()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CosineKernel;

impl Kernel for CosineKernel {
    fn similarity(&self, left: &[f64], right: &[f64]) -> f64 {
        let (mut dot, mut left_norm, mut right_norm) = (0.0, 0.0, 0.0);
        for (&left_value, &right_value) in left.iter().zip(right) {
            dot += left_value * right_value;
            left_norm += left_value * left_value;
            right_norm += right_value * right_value;
        }
        if left_norm == 0.0 || right_norm == 0.0 {
            // Two zero vectors are identical; a zero and a nonzero vector have
            // no direction in common. This keeps the Gram matrix finite.
            f64::from(left_norm == 0.0 && right_norm == 0.0)
        } else {
            dot / (left_norm * right_norm).sqrt()
        }
    }

    fn diagonal(&self, _value: &[f64]) -> f64 {
        1.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RbfKernel {
    gamma: f64,
}

impl RbfKernel {
    pub fn new(gamma: f64) -> Result<Self, Error> {
        if !gamma.is_finite() || gamma <= 0.0 {
            return Err(Error::InvalidGamma { value: gamma });
        }
        Ok(Self { gamma })
    }

    pub fn gamma(self) -> f64 {
        self.gamma
    }
}

impl Kernel for RbfKernel {
    fn similarity(&self, left: &[f64], right: &[f64]) -> f64 {
        let squared_distance: f64 = left
            .iter()
            .zip(right)
            .map(|(x, y)| {
                let difference = x - y;
                difference * difference
            })
            .sum();
        (-self.gamma * squared_distance).exp()
    }

    fn diagonal(&self, _value: &[f64]) -> f64 {
        1.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GammaPolicy {
    Fixed(f64),
    ExactMedian,
    SampledMedian { pairs: usize, seed: u64 },
}

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

fn squared_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(x, y)| {
            let difference = x - y;
            difference * difference
        })
        .sum()
}

fn median(mut values: Vec<f64>) -> Result<f64, Error> {
    if values.is_empty() {
        return Err(Error::NumericalFailure {
            context: "computing a median from an empty sample",
        });
    }
    let middle = values.len() / 2;
    let even = values.len() % 2 == 0;
    let (lower, value, _) = values.select_nth_unstable_by(middle, f64::total_cmp);
    if even {
        let lower_value = lower
            .iter()
            .max_by(|left, right| left.total_cmp(right))
            .ok_or(Error::NumericalFailure {
                context: "computing the lower half of a median",
            })?;
        Ok((*lower_value + *value) * 0.5)
    } else {
        Ok(*value)
    }
}

pub fn resolve_gamma(signal: SignalView<'_>, policy: GammaPolicy) -> Result<f64, Error> {
    let shape = signal.shape();
    if shape.n_samples < 2 {
        return match policy {
            GammaPolicy::Fixed(gamma) => Ok(RbfKernel::new(gamma)?.gamma()),
            GammaPolicy::ExactMedian | GammaPolicy::SampledMedian { .. } => Ok(1.0),
        };
    }
    let distances = match policy {
        GammaPolicy::Fixed(gamma) => return Ok(RbfKernel::new(gamma)?.gamma()),
        GammaPolicy::ExactMedian => {
            let count = shape.n_samples * (shape.n_samples - 1) / 2;
            let mut values = Vec::with_capacity(count);
            for left in 0..shape.n_samples {
                for right in left + 1..shape.n_samples {
                    values.push(squared_distance(
                        signal.row(left).ok_or(Error::NumericalFailure {
                            context: "reading a signal row for exact gamma",
                        })?,
                        signal.row(right).ok_or(Error::NumericalFailure {
                            context: "reading a signal row for exact gamma",
                        })?,
                    ));
                }
            }
            values
        }
        GammaPolicy::SampledMedian { pairs, seed } => {
            if pairs == 0 {
                return Err(Error::InvalidGammaSampleSize { value: pairs });
            }
            let mut rng = SplitMix64(seed);
            let mut values = Vec::with_capacity(pairs);
            for _ in 0..pairs {
                let left = rng.next() as usize % shape.n_samples;
                let mut right = rng.next() as usize % (shape.n_samples - 1);
                if right >= left {
                    right += 1;
                }
                values.push(squared_distance(
                    signal.row(left).ok_or(Error::NumericalFailure {
                        context: "reading a signal row for sampled gamma",
                    })?,
                    signal.row(right).ok_or(Error::NumericalFailure {
                        context: "reading a signal row for sampled gamma",
                    })?,
                ));
            }
            values
        }
    };
    let scale = median(distances)?;
    Ok(if scale > 0.0 { scale.recip() } else { 1.0 })
}

#[derive(Clone, Debug)]
pub struct FullGramPrefix<K> {
    prefix: Vec<f64>,
    diagonal_prefix: Vec<f64>,
    n_samples: usize,
    n_features: usize,
    kernel: K,
}

impl<K: Kernel> FullGramPrefix<K> {
    pub fn fit(signal: SignalView<'_>, kernel: K, max_bytes: usize) -> Result<Self, Error> {
        let shape = signal.shape();
        let side = shape
            .n_samples
            .checked_add(1)
            .ok_or(Error::NumericalFailure {
                context: "computing Gram prefix dimensions",
            })?;
        let entries = side.checked_mul(side).ok_or(Error::NumericalFailure {
            context: "computing Gram prefix dimensions",
        })?;
        let requested = entries
            .checked_add(side)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
            .ok_or(Error::NumericalFailure {
                context: "computing Gram prefix memory",
            })?;
        if requested > max_bytes {
            return Err(Error::GramMemoryLimit {
                requested,
                maximum: max_bytes,
            });
        }

        let mut prefix = vec![0.0; entries];
        let mut diagonal_prefix = vec![0.0; side];
        for row in 0..shape.n_samples {
            let row_values = signal.row(row).ok_or(Error::NumericalFailure {
                context: "reading a signal row for a Gram prefix",
            })?;
            for column in 0..shape.n_samples {
                let column_values = signal.row(column).ok_or(Error::NumericalFailure {
                    context: "reading a signal column for a Gram prefix",
                })?;
                let value = kernel.similarity(row_values, column_values);
                let target = (row + 1) * side + column + 1;
                prefix[target] =
                    value + prefix[row * side + column + 1] + prefix[(row + 1) * side + column]
                        - prefix[row * side + column];
            }
            diagonal_prefix[row + 1] =
                diagonal_prefix[row] + kernel.similarity(row_values, row_values);
        }
        Ok(Self {
            prefix,
            diagonal_prefix,
            n_samples: shape.n_samples,
            n_features: shape.n_features,
            kernel,
        })
    }

    pub fn stored_entries(&self) -> usize {
        self.prefix.len()
    }

    pub fn kernel(&self) -> &K {
        &self.kernel
    }

    fn block_sum(&self, start: usize, end: usize) -> f64 {
        let side = self.n_samples + 1;
        self.prefix[end * side + end]
            - self.prefix[start * side + end]
            - self.prefix[end * side + start]
            + self.prefix[start * side + start]
    }
}

impl<K: Kernel> SegmentCost for FullGramPrefix<K> {
    fn n_samples(&self) -> usize {
        self.n_samples
    }
    fn n_features(&self) -> usize {
        self.n_features
    }
    fn min_size(&self) -> usize {
        1
    }
    fn cost(&self, segment: Range<usize>) -> Result<f64, Error> {
        validate_segment(segment.clone(), self.n_samples, 1)?;
        let diagonal = self.diagonal_prefix[segment.end] - self.diagonal_prefix[segment.start];
        let value = diagonal - self.block_sum(segment.start, segment.end) / segment.len() as f64;
        Ok(if value < 0.0 && value > -1e-10 {
            0.0
        } else {
            value
        })
    }
}

#[derive(Clone, Debug)]
pub struct StreamingKernelCost<K> {
    values: Vec<f64>,
    diagonal_prefix: Vec<f64>,
    n_samples: usize,
    n_features: usize,
    kernel: K,
}

impl<K: Kernel> StreamingKernelCost<K> {
    pub fn fit(signal: SignalView<'_>, kernel: K) -> Self {
        let shape = signal.shape();
        let values = signal.values().to_vec();
        let mut diagonal_prefix = Vec::with_capacity(shape.n_samples + 1);
        diagonal_prefix.push(0.0);
        for sample in signal.values().chunks_exact(shape.n_features) {
            let row = diagonal_prefix.len() - 1;
            diagonal_prefix.push(diagonal_prefix[row] + kernel.similarity(sample, sample));
        }
        Self {
            values,
            diagonal_prefix,
            n_samples: shape.n_samples,
            n_features: shape.n_features,
            kernel,
        }
    }

    pub fn stored_gram_entries(&self) -> usize {
        0
    }

    fn row(&self, index: usize) -> &[f64] {
        &self.values[index * self.n_features..(index + 1) * self.n_features]
    }
}

impl<K: Kernel> SegmentCost for StreamingKernelCost<K> {
    fn n_samples(&self) -> usize {
        self.n_samples
    }
    fn n_features(&self) -> usize {
        self.n_features
    }
    fn min_size(&self) -> usize {
        1
    }
    fn cost(&self, segment: Range<usize>) -> Result<f64, Error> {
        validate_segment(segment.clone(), self.n_samples, 1)?;
        let mut block = 0.0;
        for left in segment.clone() {
            for right in segment.clone() {
                block += self.kernel.similarity(self.row(left), self.row(right));
            }
        }
        let diagonal = self.diagonal_prefix[segment.end] - self.diagonal_prefix[segment.start];
        let value = diagonal - block / segment.len() as f64;
        Ok(if value < 0.0 && value > -1e-10 {
            0.0
        } else {
            value
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KernelKind {
    Linear,
    Rbf(GammaPolicy),
    Cosine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelBackend {
    FullGram,
    Streaming,
}

#[derive(Clone, Debug)]
pub enum KernelCost {
    FullLinear(FullGramPrefix<LinearKernel>),
    FullRbf(FullGramPrefix<RbfKernel>),
    FullCosine(FullGramPrefix<CosineKernel>),
    StreamingLinear(StreamingKernelCost<LinearKernel>),
    StreamingRbf(StreamingKernelCost<RbfKernel>),
    StreamingCosine(StreamingKernelCost<CosineKernel>),
}

#[derive(Clone, Debug)]
pub enum FusedKernel {
    Linear(FusedKernelCPD<LinearKernel>),
    Rbf(FusedKernelCPD<RbfKernel>),
    Cosine(FusedKernelCPD<LinearKernel>),
}

impl FusedKernel {
    pub fn fit(
        signal: SignalView<'_>,
        kind: KernelKind,
        min_size: usize,
        jump: usize,
    ) -> Result<Self, Error> {
        Ok(match kind {
            KernelKind::Linear => {
                Self::Linear(FusedKernelCPD::fit(signal, LinearKernel, min_size, jump)?)
            }
            KernelKind::Rbf(policy) => {
                let kernel = RbfKernel::new(resolve_gamma(signal, policy)?)?;
                Self::Rbf(FusedKernelCPD::fit(signal, kernel, min_size, jump)?)
            }
            KernelKind::Cosine => {
                let shape = signal.shape();
                let mut normalized = Vec::with_capacity(shape.n_samples * (shape.n_features + 1));
                for index in 0..shape.n_samples {
                    let row = signal.row(index).ok_or(Error::NumericalFailure {
                        context: "reading a signal row for cosine normalization",
                    })?;
                    let scale = row.iter().map(|value| value.abs()).fold(0.0, f64::max);
                    if scale == 0.0 {
                        normalized.extend(std::iter::repeat_n(0.0, shape.n_features));
                        normalized.push(1.0);
                    } else {
                        let scaled_norm = row
                            .iter()
                            .map(|value| {
                                let scaled = value / scale;
                                scaled * scaled
                            })
                            .sum::<f64>()
                            .sqrt();
                        normalized.extend(row.iter().map(|value| (value / scale) / scaled_norm));
                        normalized.push(0.0);
                    }
                }
                let normalized_signal =
                    SignalView::new(&normalized, shape.n_samples, shape.n_features + 1)?;
                Self::Cosine(FusedKernelCPD::fit(
                    normalized_signal,
                    LinearKernel,
                    min_size,
                    jump,
                )?)
            }
        })
    }

    pub fn predict_changes(&self, changes: usize) -> Result<Segmentation, Error> {
        match self {
            Self::Linear(detector) => detector.predict_changes(changes),
            Self::Rbf(detector) => detector.predict_changes(changes),
            Self::Cosine(detector) => detector.predict_changes(changes),
        }
    }

    pub fn predict_penalty(&self, penalty: f64) -> Result<Segmentation, Error> {
        match self {
            Self::Linear(detector) => detector.predict_penalty(penalty),
            Self::Rbf(detector) => detector.predict_penalty(penalty),
            Self::Cosine(detector) => detector.predict_penalty(penalty),
        }
    }

    pub fn gamma(&self) -> Option<f64> {
        match self {
            Self::Rbf(detector) => Some(detector.kernel().gamma()),
            _ => None,
        }
    }
}

impl KernelCost {
    pub fn fit(
        signal: SignalView<'_>,
        kind: KernelKind,
        backend: KernelBackend,
        max_gram_bytes: usize,
    ) -> Result<Self, Error> {
        Ok(match (kind, backend) {
            (KernelKind::Linear, KernelBackend::FullGram) => {
                Self::FullLinear(FullGramPrefix::fit(signal, LinearKernel, max_gram_bytes)?)
            }
            (KernelKind::Linear, KernelBackend::Streaming) => {
                Self::StreamingLinear(StreamingKernelCost::fit(signal, LinearKernel))
            }
            (KernelKind::Cosine, KernelBackend::FullGram) => {
                Self::FullCosine(FullGramPrefix::fit(signal, CosineKernel, max_gram_bytes)?)
            }
            (KernelKind::Cosine, KernelBackend::Streaming) => {
                Self::StreamingCosine(StreamingKernelCost::fit(signal, CosineKernel))
            }
            (KernelKind::Rbf(policy), KernelBackend::FullGram) => {
                let kernel = RbfKernel::new(resolve_gamma(signal, policy)?)?;
                Self::FullRbf(FullGramPrefix::fit(signal, kernel, max_gram_bytes)?)
            }
            (KernelKind::Rbf(policy), KernelBackend::Streaming) => {
                let kernel = RbfKernel::new(resolve_gamma(signal, policy)?)?;
                Self::StreamingRbf(StreamingKernelCost::fit(signal, kernel))
            }
        })
    }

    pub fn gamma(&self) -> Option<f64> {
        match self {
            Self::FullRbf(cost) => Some(cost.kernel().gamma()),
            Self::StreamingRbf(cost) => Some(cost.kernel.gamma()),
            _ => None,
        }
    }

    pub fn stored_gram_entries(&self) -> usize {
        match self {
            Self::FullLinear(cost) => cost.stored_entries(),
            Self::FullRbf(cost) => cost.stored_entries(),
            Self::FullCosine(cost) => cost.stored_entries(),
            Self::StreamingLinear(cost) => cost.stored_gram_entries(),
            Self::StreamingRbf(cost) => cost.stored_gram_entries(),
            Self::StreamingCosine(cost) => cost.stored_gram_entries(),
        }
    }
}

impl SegmentCost for KernelCost {
    fn n_samples(&self) -> usize {
        match self {
            Self::FullLinear(cost) => cost.n_samples(),
            Self::FullRbf(cost) => cost.n_samples(),
            Self::FullCosine(cost) => cost.n_samples(),
            Self::StreamingLinear(cost) => cost.n_samples(),
            Self::StreamingRbf(cost) => cost.n_samples(),
            Self::StreamingCosine(cost) => cost.n_samples(),
        }
    }
    fn n_features(&self) -> usize {
        match self {
            Self::FullLinear(cost) => cost.n_features(),
            Self::FullRbf(cost) => cost.n_features(),
            Self::FullCosine(cost) => cost.n_features(),
            Self::StreamingLinear(cost) => cost.n_features(),
            Self::StreamingRbf(cost) => cost.n_features(),
            Self::StreamingCosine(cost) => cost.n_features(),
        }
    }
    fn min_size(&self) -> usize {
        1
    }
    fn cost(&self, segment: Range<usize>) -> Result<f64, Error> {
        match self {
            Self::FullLinear(cost) => cost.cost(segment),
            Self::FullRbf(cost) => cost.cost(segment),
            Self::FullCosine(cost) => cost.cost(segment),
            Self::StreamingLinear(cost) => cost.cost(segment),
            Self::StreamingRbf(cost) => cost.cost(segment),
            Self::StreamingCosine(cost) => cost.cost(segment),
        }
    }
}

/// Exact kernel detector using either an O(n²) prefix backend or a
/// Gram-free streaming backend.
#[derive(Clone, Debug)]
pub struct KernelCPD {
    cost: KernelCost,
    dynp: Dynp,
    pelt: Pelt,
}

impl KernelCPD {
    pub fn fit(
        signal: SignalView<'_>,
        kind: KernelKind,
        backend: KernelBackend,
        min_size: usize,
        jump: usize,
        max_gram_bytes: usize,
    ) -> Result<Self, Error> {
        Ok(Self {
            cost: KernelCost::fit(signal, kind, backend, max_gram_bytes)?,
            dynp: Dynp::new(min_size, jump)?,
            pelt: Pelt::new(min_size, jump)?,
        })
    }

    pub fn predict_changes(&self, changes: usize) -> Result<Segmentation, Error> {
        self.dynp.predict_changes(&self.cost, changes)
    }

    pub fn predict_penalty(&self, penalty: f64) -> Result<Segmentation, Error> {
        self.pelt.predict_penalty(&self.cost, penalty)
    }

    pub fn cost(&self) -> &KernelCost {
        &self.cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::{best_fixed_changes, best_penalized};

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
}
