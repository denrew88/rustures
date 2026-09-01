use std::ops::Range;

use crate::{validate_segment, Error, SegmentCost, SignalShape, SignalView};

#[derive(Clone, Debug)]
pub struct CostL1 {
    shape: SignalShape,
    values: Vec<f64>,
}

impl CostL1 {
    pub fn fit(signal: SignalView<'_>) -> Self {
        Self {
            shape: signal.shape(),
            values: signal.values().to_vec(),
        }
    }
}

impl SegmentCost for CostL1 {
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
        validate_segment(segment.clone(), self.n_samples(), 1)?;
        let mut total = 0.0;
        let mut column = Vec::with_capacity(segment.len());
        for feature in 0..self.n_features() {
            column.clear();
            column.extend(
                segment
                    .clone()
                    .map(|row| self.values[row * self.n_features() + feature]),
            );
            column.sort_unstable_by(f64::total_cmp);
            let median = column[(column.len() - 1) / 2];
            total += column
                .iter()
                .map(|value| (value - median).abs())
                .sum::<f64>();
        }
        if total.is_finite() {
            Ok(total)
        } else {
            Err(Error::NonFiniteObjective { value: total })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_componentwise_medians() {
        let values = [0., 10., 2., 8., 100., 6.];
        let cost = CostL1::fit(SignalView::new(&values, 3, 2).unwrap());
        assert_eq!(cost.cost(0..3).unwrap(), 104.0);
    }
}
