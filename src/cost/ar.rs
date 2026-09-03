use std::ops::Range;

use super::linear::RegressionData;
use crate::{validate_segment, Error, SegmentCost, SignalShape, SignalView};

#[derive(Clone, Debug)]
pub struct CostAR {
    shape: SignalShape,
    regression: RegressionData,
    order: usize,
    min_size: usize,
    boundary: ARBoundary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ARBoundary {
    /// Reuse lagged observations from before a candidate segment. Missing
    /// lags at the beginning of the full signal are zero-filled.
    Compatibility,
    /// Drop the first `order` responses inside every candidate segment.
    SegmentLocal,
}

impl CostAR {
    pub fn fit(signal: SignalView<'_>, order: usize) -> Result<Self, Error> {
        Self::fit_with_boundary(signal, order, ARBoundary::Compatibility)
    }

    pub fn fit_segment_local(signal: SignalView<'_>, order: usize) -> Result<Self, Error> {
        Self::fit_with_boundary(signal, order, ARBoundary::SegmentLocal)
    }

    pub fn fit_with_boundary(
        signal: SignalView<'_>,
        order: usize,
        boundary: ARBoundary,
    ) -> Result<Self, Error> {
        if order == 0 {
            return Err(Error::InvalidOrder { value: order });
        }
        let shape = signal.shape();
        let min_size = order
            .checked_add(1)
            .ok_or(Error::NumericalFailure {
                context: "computing autoregressive minimum segment length",
            })?
            .max(5);
        let predictors =
            1usize
                .checked_add(order.checked_mul(shape.n_features).ok_or(
                    Error::NumericalFailure {
                        context: "computing autoregressive predictor count",
                    },
                )?)
                .ok_or(Error::NumericalFailure {
                    context: "computing autoregressive predictor count",
                })?;
        let design_len =
            shape
                .n_samples
                .checked_mul(predictors)
                .ok_or(Error::NumericalFailure {
                    context: "allocating autoregressive design storage",
                })?;
        let mut design = Vec::new();
        design
            .try_reserve_exact(design_len)
            .map_err(|_| Error::NumericalFailure {
                context: "allocating autoregressive design storage",
            })?;
        for sample in 0..shape.n_samples {
            design.push(1.0);
            for lag in 1..=order {
                for feature in 0..shape.n_features {
                    let value = sample.checked_sub(lag).map_or(0.0, |lagged_sample| {
                        signal.values()[lagged_sample * shape.n_features + feature]
                    });
                    design.push(value);
                }
            }
        }
        Ok(Self {
            shape,
            regression: RegressionData::new(
                design,
                signal.values().to_vec(),
                shape.n_samples,
                predictors,
                shape.n_features,
            ),
            order,
            min_size,
            boundary,
        })
    }
    pub fn order(&self) -> usize {
        self.order
    }
    pub fn boundary(&self) -> ARBoundary {
        self.boundary
    }
}

impl SegmentCost for CostAR {
    fn n_samples(&self) -> usize {
        self.shape.n_samples
    }
    fn n_features(&self) -> usize {
        self.shape.n_features
    }
    fn min_size(&self) -> usize {
        self.min_size
    }

    fn cost(&self, segment: Range<usize>) -> Result<f64, Error> {
        validate_segment(segment.clone(), self.n_samples(), self.min_size)?;
        let local_offset = match self.boundary {
            ARBoundary::Compatibility => 0,
            ARBoundary::SegmentLocal => self.order,
        };
        self.regression
            .residual(segment.start + local_offset..segment.end)
    }

    fn costs_ending_at(
        &self,
        starts: &[usize],
        end: usize,
        output: &mut Vec<f64>,
    ) -> Result<(), Error> {
        let local_offset = match self.boundary {
            ARBoundary::Compatibility => 0,
            ARBoundary::SegmentLocal => self.order,
        };
        self.regression
            .costs_ending_at(starts, end, local_offset, self.min_size, output)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cost/ar.rs"]
mod tests;
