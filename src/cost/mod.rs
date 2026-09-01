use std::ops::Range;

use crate::Error;

mod ar;
mod clinear;
mod l1;
mod l2;
mod linear;
mod mahalanobis;
mod model;
mod normal;
mod rank;
mod scatter;

pub use ar::{ARBoundary, CostAR};
pub use clinear::CostCLinear;
pub use l1::CostL1;
pub use l2::CostL2;
pub use linear::CostLinear;
pub use mahalanobis::CostMahalanobis;
pub use model::{CostModel, CostSpec};
pub use normal::CostNormal;
pub use rank::CostRank;

pub trait SegmentCost: Send + Sync {
    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize;
    fn min_size(&self) -> usize;
    fn cost(&self, segment: Range<usize>) -> Result<f64, Error>;

    /// Evaluate several segments sharing the same exclusive endpoint.
    ///
    /// The default implementation preserves the scalar cost contract. Costs
    /// with an endpoint-sweep implementation can override this method to reuse
    /// work without retaining an `O(n^2)` segment-cost table.
    fn costs_ending_at(
        &self,
        starts: &[usize],
        end: usize,
        output: &mut Vec<f64>,
    ) -> Result<(), Error> {
        output.clear();
        output
            .try_reserve(starts.len())
            .map_err(|_| Error::NumericalFailure {
                context: "allocating an endpoint cost batch",
            })?;
        for &start in starts {
            output.push(self.cost(start..end)?);
        }
        Ok(())
    }

    fn pelt_pruning_constant(&self) -> Option<f64> {
        None
    }
}
