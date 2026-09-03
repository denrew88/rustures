use std::ops::Range;

use super::scatter::ScatterStats;
use crate::{Error, SegmentCost, SignalView};

#[derive(Clone, Debug)]
pub struct CostNormal {
    stats: ScatterStats,
    ridge: f64,
}

impl CostNormal {
    pub fn fit(signal: SignalView<'_>, ridge: f64) -> Result<Self, Error> {
        if !ridge.is_finite() || ridge <= 0.0 {
            return Err(Error::InvalidRidge { value: ridge });
        }
        Ok(Self {
            stats: ScatterStats::fit(signal)?,
            ridge,
        })
    }
    pub fn ridge(&self) -> f64 {
        self.ridge
    }
}

impl SegmentCost for CostNormal {
    fn n_samples(&self) -> usize {
        self.stats.shape.n_samples
    }
    fn n_features(&self) -> usize {
        self.stats.shape.n_features
    }
    fn min_size(&self) -> usize {
        2
    }

    fn cost(&self, segment: Range<usize>) -> Result<f64, Error> {
        let length = segment.len() as f64;
        let mut covariance = self.stats.scatter(segment, 2)? / (length - 1.0);
        for index in 0..self.n_features() {
            covariance[(index, index)] += self.ridge;
        }
        let cholesky = covariance.cholesky().ok_or(Error::NumericalFailure {
            context: "factoring a regularized covariance matrix",
        })?;
        let log_determinant = 2.0
            * cholesky
                .l()
                .diagonal()
                .iter()
                .map(|value| value.ln())
                .sum::<f64>();
        let result = length * log_determinant;
        if result.is_finite() {
            Ok(result)
        } else {
            Err(Error::NonFiniteObjective { value: result })
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cost/normal.rs"]
mod tests;
