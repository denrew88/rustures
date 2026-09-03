use std::ops::Range;

use crate::{validate_segment, Error, SegmentCost, SignalShape, SignalView};

#[derive(Clone, Debug)]
pub struct CostCLinear {
    shape: SignalShape,
    values: Vec<f64>,
}

impl CostCLinear {
    pub fn fit(signal: SignalView<'_>) -> Self {
        Self {
            shape: signal.shape(),
            values: signal.values().to_vec(),
        }
    }
}

impl SegmentCost for CostCLinear {
    fn n_samples(&self) -> usize {
        self.shape.n_samples
    }
    fn n_features(&self) -> usize {
        self.shape.n_features
    }
    fn min_size(&self) -> usize {
        3
    }

    fn cost(&self, segment: Range<usize>) -> Result<f64, Error> {
        validate_segment(segment.clone(), self.n_samples(), 3)?;
        let anchor = segment.start.saturating_sub(1);
        let denominator = (segment.end - 1 - anchor) as f64;
        let mut total = 0.0;
        for feature in 0..self.n_features() {
            let left = self.values[anchor * self.n_features() + feature];
            let right = self.values[(segment.end - 1) * self.n_features() + feature];
            for row in segment.clone() {
                let fraction = (row - anchor) as f64 / denominator;
                let fitted = left + fraction * (right - left);
                let residual = self.values[row * self.n_features() + feature] - fitted;
                total += residual * residual;
            }
        }
        if total.is_finite() {
            Ok(total)
        } else {
            Err(Error::NonFiniteObjective { value: total })
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cost/clinear.rs"]
mod tests;
