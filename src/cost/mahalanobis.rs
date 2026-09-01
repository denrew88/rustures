use std::ops::Range;

use nalgebra::DMatrix;

use super::scatter::{clamp_nonnegative, validate_psd, ScatterStats};
use crate::{Error, SegmentCost, SignalView};

#[derive(Clone, Debug)]
pub struct CostMahalanobis {
    stats: ScatterStats,
    metric: DMatrix<f64>,
}

impl CostMahalanobis {
    pub fn fit(
        signal: SignalView<'_>,
        metric: Vec<f64>,
        rows: usize,
        columns: usize,
    ) -> Result<Self, Error> {
        let expected = signal.shape().n_features;
        if rows != expected || columns != expected || metric.len() != rows.saturating_mul(columns) {
            return Err(Error::InvalidMetricShape {
                expected,
                rows,
                columns,
            });
        }
        let metric = DMatrix::from_row_slice(rows, columns, &metric);
        validate_psd(&metric)?;
        Ok(Self {
            stats: ScatterStats::fit(signal)?,
            metric,
        })
    }
    pub fn identity(signal: SignalView<'_>) -> Result<Self, Error> {
        let d = signal.shape().n_features;
        Self::fit(signal, DMatrix::identity(d, d).as_slice().to_vec(), d, d)
    }
}

impl SegmentCost for CostMahalanobis {
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
        let scatter = self.stats.scatter(segment, 1)?;
        let value = (&self.metric * &scatter).trace();
        let scale = self.metric.norm() * scatter.norm();
        clamp_nonnegative(value, scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagonal_metric_weights_feature_scatter() {
        let values = [0., 0., 2., 1.];
        let signal = SignalView::new(&values, 2, 2).unwrap();
        let cost = CostMahalanobis::fit(signal, vec![2., 0., 0., 3.], 2, 2).unwrap();
        assert_eq!(cost.cost(0..2).unwrap(), 5.5);
    }
    #[test]
    fn rejects_indefinite_metric() {
        let signal = SignalView::new(&[0., 0., 1., 1.], 2, 2).unwrap();
        assert!(matches!(
            CostMahalanobis::fit(signal, vec![1., 2., 2., 1.], 2, 2),
            Err(Error::NonPositiveSemidefiniteMetric)
        ));
    }
}
