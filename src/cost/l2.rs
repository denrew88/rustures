use std::ops::Range;

use crate::{validate_breakpoints, validate_segment, Error, SegmentCost, SignalShape, SignalView};

const MIN_SIZE: usize = 1;
const ROUNDOFF_FACTOR: f64 = 64.0;

#[derive(Clone, Debug)]
pub struct CostL2 {
    shape: SignalShape,
    offsets: Vec<f64>,
    prefix_sum: Vec<f64>,
    prefix_square_sum: Vec<f64>,
}

impl CostL2 {
    pub fn fit(signal: SignalView<'_>) -> Result<Self, Error> {
        Self::from_values(signal.values().iter().copied(), signal.shape())
    }

    pub fn from_values(
        values: impl IntoIterator<Item = f64>,
        shape: SignalShape,
    ) -> Result<Self, Error> {
        let expected =
            shape
                .n_samples
                .checked_mul(shape.n_features)
                .ok_or(Error::NumericalFailure {
                    context: "computing signal storage size",
                })?;
        if shape.n_samples == 0 || shape.n_features == 0 {
            return Err(Error::EmptySignal);
        }

        let prefix_rows = shape
            .n_samples
            .checked_add(1)
            .ok_or(Error::NumericalFailure {
                context: "allocating L2 prefix statistics",
            })?;
        let prefix_len =
            prefix_rows
                .checked_mul(shape.n_features)
                .ok_or(Error::NumericalFailure {
                    context: "allocating L2 prefix statistics",
                })?;
        let mut offsets = vec![0.0; shape.n_features];
        let mut prefix_sum = vec![0.0; prefix_len];
        let mut prefix_square_sum = vec![0.0; prefix_len];
        let mut sums = vec![0.0; shape.n_features];
        let mut sum_corrections = vec![0.0; shape.n_features];
        let mut square_sums = vec![0.0; shape.n_features];
        let mut square_corrections = vec![0.0; shape.n_features];
        let mut values = values.into_iter();

        for flat_index in 0..expected {
            let value = values.next().ok_or(Error::DimensionMismatch {
                expected,
                actual: flat_index,
            })?;
            let feature = flat_index % shape.n_features;
            let row = flat_index / shape.n_features;
            if !value.is_finite() {
                return Err(Error::NonFiniteInput {
                    row,
                    column: feature,
                });
            }
            if row == 0 {
                offsets[feature] = value;
            }
            let centered = value - offsets[feature];
            let square = centered * centered;
            if !square.is_finite() {
                return Err(Error::NumericalFailure {
                    context: "forming squared centered observations",
                });
            }
            compensated_add(&mut sums[feature], &mut sum_corrections[feature], centered);
            compensated_add(
                &mut square_sums[feature],
                &mut square_corrections[feature],
                square,
            );
            let prefix_index = (row + 1) * shape.n_features + feature;
            prefix_sum[prefix_index] = sums[feature];
            prefix_square_sum[prefix_index] = square_sums[feature];
        }

        if values.next().is_some() {
            let actual = expected.saturating_add(1).saturating_add(values.count());
            return Err(Error::DimensionMismatch { expected, actual });
        }

        Ok(Self {
            shape,
            offsets,
            prefix_sum,
            prefix_square_sum,
        })
    }

    pub fn offsets(&self) -> &[f64] {
        &self.offsets
    }

    pub fn sum_of_costs(&self, breakpoints: &[usize]) -> Result<f64, Error> {
        validate_breakpoints(breakpoints, self.n_samples(), self.min_size())?;
        let mut start = 0;
        let mut total = 0.0;
        let mut correction = 0.0;
        for &end in breakpoints {
            compensated_add(&mut total, &mut correction, self.cost(start..end)?);
            start = end;
        }
        if !total.is_finite() {
            return Err(Error::NonFiniteObjective { value: total });
        }
        Ok(total)
    }
}

impl SegmentCost for CostL2 {
    fn n_samples(&self) -> usize {
        self.shape.n_samples
    }

    fn n_features(&self) -> usize {
        self.shape.n_features
    }

    fn min_size(&self) -> usize {
        MIN_SIZE
    }

    fn cost(&self, segment: Range<usize>) -> Result<f64, Error> {
        validate_segment(segment.clone(), self.n_samples(), self.min_size())?;
        let length = (segment.end - segment.start) as f64;
        let mut total = 0.0;
        let mut correction = 0.0;

        for feature in 0..self.n_features() {
            let start_index = segment.start * self.n_features() + feature;
            let end_index = segment.end * self.n_features() + feature;
            let sum = self.prefix_sum[end_index] - self.prefix_sum[start_index];
            let start_square_sum = self.prefix_square_sum[start_index];
            let end_square_sum = self.prefix_square_sum[end_index];
            let square_sum = end_square_sum - start_square_sum;
            let mean_term = (sum / length) * sum;
            let raw = square_sum - mean_term;
            let roundoff_scale = start_square_sum
                .abs()
                .max(end_square_sum.abs())
                .max(mean_term.abs());
            let value = clamp_roundoff(raw, roundoff_scale)?;
            compensated_add(&mut total, &mut correction, value);
        }

        if !total.is_finite() {
            return Err(Error::NonFiniteObjective { value: total });
        }
        Ok(total)
    }
}

fn compensated_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let adjusted = value - *correction;
    let next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
}

fn clamp_roundoff(value: f64, scale: f64) -> Result<f64, Error> {
    if !value.is_finite() {
        return Err(Error::NonFiniteObjective { value });
    }
    if value >= 0.0 {
        return Ok(value);
    }
    let tolerance = ROUNDOFF_FACTOR * f64::EPSILON * scale.max(1.0);
    if value >= -tolerance {
        Ok(0.0)
    } else {
        Err(Error::NumericalFailure {
            context: "evaluating an L2 segment cost",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_cost(values: &[f64], n_features: usize, segment: Range<usize>) -> f64 {
        let length = (segment.end - segment.start) as f64;
        (0..n_features)
            .map(|feature| {
                let mean = (segment.clone())
                    .map(|row| values[row * n_features + feature])
                    .sum::<f64>()
                    / length;
                (segment.clone())
                    .map(|row| {
                        let difference = values[row * n_features + feature] - mean;
                        difference * difference
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    #[test]
    fn scalar_cost_matches_direct_variance() {
        let values = [1.0, 2.0, 5.0, 6.0];
        let cost = CostL2::fit(SignalView::new(&values, 4, 1).unwrap()).unwrap();
        assert_eq!(cost.cost(0..4).unwrap(), 17.0);
        assert_eq!(cost.cost(1..3).unwrap(), 4.5);
    }

    #[test]
    fn multivariate_subranges_match_direct_calculation() {
        let values = [1.0, 10.0, 2.0, 8.0, 5.0, 7.0, 9.0, 3.0, 10.0, 2.0];
        let cost = CostL2::fit(SignalView::new(&values, 5, 2).unwrap()).unwrap();
        for start in 0..5 {
            for end in start + 1..=5 {
                let expected = direct_cost(&values, 2, start..end);
                let actual = cost.cost(start..end).unwrap();
                assert!((actual - expected).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn constant_and_large_offset_signals_are_stable() {
        let constant = [7.0; 12];
        let cost = CostL2::fit(SignalView::new(&constant, 6, 2).unwrap()).unwrap();
        assert_eq!(cost.cost(0..6).unwrap(), 0.0);

        let large = [1.0e12, 1.0e12 + 2.0, 1.0e12 + 4.0];
        let cost = CostL2::fit(SignalView::new(&large, 3, 1).unwrap()).unwrap();
        assert_eq!(cost.cost(0..3).unwrap(), 8.0);
        assert_eq!(cost.offsets(), &[1.0e12]);
    }

    #[test]
    fn roundoff_policy_clamps_only_small_negative_values() {
        assert_eq!(clamp_roundoff(-f64::EPSILON, 1.0), Ok(0.0));
        assert!(matches!(
            clamp_roundoff(-1.0, 1.0),
            Err(Error::NumericalFailure { .. })
        ));
    }

    #[test]
    fn validates_ranges_and_partition_costs() {
        let values = [0.0, 0.0, 10.0, 10.0];
        let cost = CostL2::fit(SignalView::new(&values, 4, 1).unwrap()).unwrap();
        assert_eq!(cost.sum_of_costs(&[2, 4]).unwrap(), 0.0);
        assert!(matches!(cost.cost(4..4), Err(Error::InvalidRange { .. })));
    }

    #[test]
    fn late_short_segments_remain_nonnegative() {
        let values: Vec<f64> = (0..1_000)
            .map(|index| {
                let level = (index / 50 % 5) as f64 * 4.0;
                level + ((index * 37 % 101) as f64 / 101.0 - 0.5)
            })
            .collect();
        let cost = CostL2::fit(SignalView::new(&values, values.len(), 1).unwrap()).unwrap();
        for start in 900..1_000 {
            for end in start + 1..=1_000 {
                let value = cost.cost(start..end).unwrap();
                assert!(value.is_finite() && value >= 0.0);
            }
        }
    }
}
