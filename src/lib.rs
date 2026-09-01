mod core;
mod cost;
mod datasets;
mod kernel;
mod metrics;
mod python;
mod search;

#[cfg(test)]
mod testing;

pub use core::error::Error;
pub use core::segmentation::{
    candidate_is_better, objective_values_tied, partition_cost, validate_breakpoints,
    validate_budget, validate_jump, validate_min_size, validate_penalty, validate_segment,
    Detector, DetectorCapabilities, SearchGrid, Segmentation, Stop,
};
pub use core::signal::{validate_finite, validate_signal_shape, SignalShape, SignalView};
pub use cost::{
    ARBoundary, CostAR, CostCLinear, CostL1, CostL2, CostLinear, CostMahalanobis, CostModel,
    CostNormal, CostRank, CostSpec, SegmentCost,
};
pub use kernel::{
    resolve_gamma, CosineKernel, FullGramPrefix, FusedKernel, GammaPolicy, Kernel, KernelBackend,
    KernelCPD, KernelCost, KernelKind, LinearKernel, RbfKernel, StreamingKernelCost,
};
pub use metrics::{hausdorff, precision_recall, rand_index};
pub use search::{Binseg, BottomUp, Dynp, FusedKernelCPD, L1Potts, Pelt, Window};

#[cfg(test)]
pub(crate) use testing::oracle;
