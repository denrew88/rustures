use std::ops::Range;

use crate::Error;

mod l2;

pub use l2::CostL2;

pub trait SegmentCost: Send + Sync {
    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize;
    fn min_size(&self) -> usize;
    fn cost(&self, segment: Range<usize>) -> Result<f64, Error>;
}
