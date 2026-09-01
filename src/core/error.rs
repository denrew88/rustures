use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum Error {
    #[error("signal must contain at least one sample and one feature")]
    EmptySignal,

    #[error("signal must be one- or two-dimensional, got {ndim} dimensions")]
    InvalidSignalDimension { ndim: usize },

    #[error("signal contains a non-finite value at row {row}, column {column}")]
    NonFiniteInput { row: usize, column: usize },

    #[error("signal storage has length {actual}, expected {expected}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("response and design have different sample counts: {response} and {design}")]
    SampleCountMismatch { response: usize, design: usize },

    #[error("invalid segment [{start}, {end}) for a signal of length {n_samples}")]
    InvalidRange {
        start: usize,
        end: usize,
        n_samples: usize,
    },

    #[error("segment [{start}, {end}) has length {length}, below minimum {minimum}")]
    SegmentTooShort {
        start: usize,
        end: usize,
        length: usize,
        minimum: usize,
    },

    #[error("min_size must be positive, got {value}")]
    InvalidMinSize { value: usize },

    #[error("min_size {requested} is below the cost function minimum {minimum}")]
    MinSizeBelowCost { requested: usize, minimum: usize },

    #[error("jump must be positive, got {value}")]
    InvalidJump { value: usize },

    #[error("penalty must be finite and positive, got {value}")]
    InvalidPenalty { value: f64 },

    #[error("reconstruction budget must be finite and nonnegative, got {value}")]
    InvalidBudget { value: f64 },

    #[error("breakpoints must not be empty")]
    EmptyBreakpoints,

    #[error("breakpoint at position {position} is {value}, but breakpoints must be strictly increasing and no larger than {n_samples}")]
    InvalidBreakpoint {
        position: usize,
        value: usize,
        n_samples: usize,
    },

    #[error("last breakpoint must equal n_samples ({n_samples}), got {actual}")]
    MissingTerminalBreakpoint { actual: usize, n_samples: usize },

    #[error("exhaustive test oracle supports at most {maximum} samples, got {actual}")]
    ExhaustiveLimitExceeded { actual: usize, maximum: usize },

    #[error("objective value must be finite, got {value}")]
    NonFiniteObjective { value: f64 },

    #[error("cannot segment {n_samples} samples into {changes} changes with min_size={min_size} and jump={jump}")]
    InfeasibleSegmentation {
        n_samples: usize,
        changes: usize,
        min_size: usize,
        jump: usize,
    },

    #[error("{detector} does not support the {rule} stopping rule")]
    UnsupportedStoppingRule {
        detector: &'static str,
        rule: &'static str,
    },

    #[error("exactly one stopping rule must be provided")]
    InvalidStoppingRules,

    #[error(
        "breakpoint sequences describe different sample lengths: expected {expected}, got {actual}"
    )]
    BreakpointLengthMismatch { expected: usize, actual: usize },

    #[error("margin must be positive, got {value}")]
    InvalidMargin { value: usize },

    #[error("invalid synthetic dataset parameters")]
    InvalidDatasetParameters,

    #[error("gamma must be finite and positive, got {value}")]
    InvalidGamma { value: f64 },

    #[error("sampled median requires at least one pair, got {value}")]
    InvalidGammaSampleSize { value: usize },

    #[error("Gram prefix requires {requested} bytes, above configured limit {maximum}")]
    GramMemoryLimit { requested: usize, maximum: usize },

    #[error("Dynp prediction requires {requested} bytes, above configured limit {maximum}")]
    DynpMemoryLimit { requested: usize, maximum: usize },

    #[error("memory limit must be positive, got {value}")]
    InvalidMemoryLimit { value: usize },

    #[error("memory allocation failed while {context}")]
    AllocationFailure { context: &'static str },

    #[error("unsupported model {model:?}; supported models: l2, l1, rank, normal, linear, ar, clinear, mahalanobis")]
    UnsupportedModel { model: String },

    #[error("ridge must be finite and positive, got {value}")]
    InvalidRidge { value: f64 },

    #[error("autoregressive order must be positive, got {value}")]
    InvalidOrder { value: usize },

    #[error("{model} requires at least {minimum} features, got {actual}")]
    InsufficientFeatures {
        model: &'static str,
        minimum: usize,
        actual: usize,
    },

    #[error("{model} requires a scalar signal with exactly one feature, got {actual}")]
    ScalarSignalRequired { model: &'static str, actual: usize },

    #[error("weights have length {actual}, expected {expected}")]
    InvalidWeightsLength { expected: usize, actual: usize },

    #[error("weight at position {position} must be finite and nonnegative, got {value}")]
    InvalidWeight { position: usize, value: f64 },

    #[error("Mahalanobis metric must be a {expected}x{expected} matrix, got {rows}x{columns}")]
    InvalidMetricShape {
        expected: usize,
        rows: usize,
        columns: usize,
    },

    #[error("Mahalanobis metric must be finite, symmetric, and positive semidefinite")]
    NonPositiveSemidefiniteMetric,

    #[error("unsupported kernel {kernel:?}; supported kernels: linear, rbf, cosine")]
    UnsupportedKernel { kernel: String },

    #[error("unsupported kernel backend {backend:?}; supported backends: fused, full, streaming")]
    UnsupportedKernelBackend { backend: String },

    #[error("unsupported gamma policy {policy:?}; supported policies: exact, sampled")]
    UnsupportedGammaPolicy { policy: String },

    #[error("{object} must be fitted before this operation")]
    NotFitted { object: &'static str },

    #[error("numerical failure while {context}")]
    NumericalFailure { context: &'static str },
}
