use std::ops::Range;

use nalgebra::{DMatrix, DVector};

use super::scatter::symmetric_pseudoinverse;
use crate::{validate_segment, Error, SegmentCost, SignalShape, SignalView};

#[derive(Clone, Debug)]
pub struct CostRank {
    shape: SignalShape,
    prefix: Vec<f64>,
    inverse_covariance: DMatrix<f64>,
}

impl CostRank {
    pub fn fit(signal: SignalView<'_>) -> Result<Self, Error> {
        let shape = signal.shape();
        let n = shape.n_samples;
        let d = shape.n_features;
        let mut ranks = vec![0.0; n * d];
        for feature in 0..d {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by(|&left, &right| {
                signal.values()[left * d + feature].total_cmp(&signal.values()[right * d + feature])
            });
            let mut start = 0;
            while start < n {
                let value = signal.values()[order[start] * d + feature];
                let mut end = start + 1;
                while end < n && signal.values()[order[end] * d + feature] == value {
                    end += 1;
                }
                let average_rank = 0.5 * ((start + 1 + end) as f64) - 0.5 * (n + 1) as f64;
                for &row in &order[start..end] {
                    ranks[row * d + feature] = average_rank;
                }
                start = end;
            }
        }
        let mut covariance = DMatrix::zeros(d, d);
        for row in 0..n {
            for left in 0..d {
                for right in 0..d {
                    covariance[(left, right)] +=
                        ranks[row * d + left] * ranks[row * d + right] / n as f64;
                }
            }
        }
        let inverse_covariance = symmetric_pseudoinverse(covariance)?;
        let mut prefix = vec![0.0; (n + 1) * d];
        for row in 0..n {
            for feature in 0..d {
                prefix[(row + 1) * d + feature] =
                    prefix[row * d + feature] + ranks[row * d + feature];
            }
        }
        Ok(Self {
            shape,
            prefix,
            inverse_covariance,
        })
    }
}

impl SegmentCost for CostRank {
    fn n_samples(&self) -> usize {
        self.shape.n_samples
    }
    fn n_features(&self) -> usize {
        self.shape.n_features
    }
    fn min_size(&self) -> usize {
        2
    }

    fn cost(&self, segment: Range<usize>) -> Result<f64, Error> {
        validate_segment(segment.clone(), self.n_samples(), 2)?;
        let length = segment.len() as f64;
        let mean = DVector::from_fn(self.n_features(), |feature, _| {
            (self.prefix[segment.end * self.n_features() + feature]
                - self.prefix[segment.start * self.n_features() + feature])
                / length
        });
        let value = -length * mean.dot(&(&self.inverse_covariance * &mean));
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::NonFiniteObjective { value })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rank_cost_detects_distribution_shift() {
        let values = [0., 1., 2., 10., 11., 12.];
        let cost = CostRank::fit(SignalView::new(&values, 6, 1).unwrap()).unwrap();
        assert!(cost.cost(0..3).unwrap() < cost.cost(1..5).unwrap());
    }

    #[test]
    fn all_tied_ranks_use_a_zero_pseudoinverse_without_nan() {
        let values = [2.0; 8];
        let cost = CostRank::fit(SignalView::new(&values, 8, 1).unwrap()).unwrap();
        assert_eq!(cost.cost(0..4).unwrap(), 0.0);
    }
}
