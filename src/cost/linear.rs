use std::ops::Range;

use nalgebra::DMatrix;

use crate::{validate_segment, Error, SegmentCost, SignalShape, SignalView};

const FAST_PATH_DIAGONAL_RATIO: f64 = 1.0e-6;
const FAST_PATH_CANCELLATION_RATIO: f64 = 1.0e-10;

pub(super) fn least_squares_residual(
    design: DMatrix<f64>,
    response: DMatrix<f64>,
) -> Result<f64, Error> {
    let coefficients = design
        .clone()
        .svd(true, true)
        .solve(&response, 1.0e-12)
        .map_err(|_| Error::NumericalFailure {
            context: "solving a least-squares segment model",
        })?;
    let residual = response - design * coefficients;
    let value = residual.iter().map(|entry| entry * entry).sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::NonFiniteObjective { value })
    }
}

#[derive(Clone, Debug)]
pub(super) struct RegressionData {
    design: Vec<f64>,
    response: Vec<f64>,
    n_samples: usize,
    n_predictors: usize,
    n_responses: usize,
}

impl RegressionData {
    pub(super) fn new(
        design: Vec<f64>,
        response: Vec<f64>,
        n_samples: usize,
        n_predictors: usize,
        n_responses: usize,
    ) -> Self {
        Self {
            design,
            response,
            n_samples,
            n_predictors,
            n_responses,
        }
    }

    pub(super) fn residual(&self, rows: Range<usize>) -> Result<f64, Error> {
        let row_count = rows.len();
        let design = DMatrix::from_fn(row_count, self.n_predictors, |row, column| {
            self.design[(rows.start + row) * self.n_predictors + column]
        });
        let response = DMatrix::from_fn(row_count, self.n_responses, |row, column| {
            self.response[(rows.start + row) * self.n_responses + column]
        });
        least_squares_residual(design, response)
    }

    pub(super) fn costs_ending_at(
        &self,
        starts: &[usize],
        end: usize,
        response_offset: usize,
        minimum_segment_length: usize,
        output: &mut Vec<f64>,
    ) -> Result<(), Error> {
        if starts.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::NumericalFailure {
                context: "evaluating an unordered endpoint cost batch",
            });
        }
        output.clear();
        output
            .try_reserve(starts.len())
            .map_err(|_| Error::NumericalFailure {
                context: "allocating a regression endpoint cost batch",
            })?;

        let mut accumulator = RegressionAccumulator::new(self.n_predictors, self.n_responses);
        let mut accumulated_from = end;
        for &start in starts.iter().rev() {
            validate_segment(start..end, self.n_samples, minimum_segment_length)?;
            let row_start = start
                .checked_add(response_offset)
                .ok_or(Error::NumericalFailure {
                    context: "computing a regression response offset",
                })?;
            if row_start > accumulated_from {
                return Err(Error::NumericalFailure {
                    context: "evaluating an invalid regression endpoint batch",
                });
            }
            for row in (row_start..accumulated_from).rev() {
                accumulator.add_row(self, row);
            }
            accumulated_from = row_start;

            let value = match accumulator.fast_residual() {
                Some(value) => value,
                None => self.residual(row_start..end)?,
            };
            output.push(value);
        }
        output.reverse();
        Ok(())
    }
}

struct RegressionAccumulator {
    cross_predictors: Vec<f64>,
    cross_response: Vec<f64>,
    response_square_sum: f64,
    n_predictors: usize,
    n_responses: usize,
}

impl RegressionAccumulator {
    fn new(n_predictors: usize, n_responses: usize) -> Self {
        Self {
            cross_predictors: vec![0.0; n_predictors * n_predictors],
            cross_response: vec![0.0; n_predictors * n_responses],
            response_square_sum: 0.0,
            n_predictors,
            n_responses,
        }
    }

    fn add_row(&mut self, data: &RegressionData, row: usize) {
        let design_start = row * self.n_predictors;
        let response_start = row * self.n_responses;
        let design = &data.design[design_start..design_start + self.n_predictors];
        let response = &data.response[response_start..response_start + self.n_responses];

        for left in 0..self.n_predictors {
            for right in 0..self.n_predictors {
                self.cross_predictors[left * self.n_predictors + right] +=
                    design[left] * design[right];
            }
            for (column, &response_value) in response.iter().enumerate() {
                self.cross_response[left * self.n_responses + column] +=
                    design[left] * response_value;
            }
        }
        self.response_square_sum += response.iter().map(|value| value * value).sum::<f64>();
    }

    fn fast_residual(&self) -> Option<f64> {
        if !self.response_square_sum.is_finite()
            || self.cross_predictors.iter().any(|value| !value.is_finite())
            || self.cross_response.iter().any(|value| !value.is_finite())
        {
            return None;
        }

        let gram =
            DMatrix::from_row_slice(self.n_predictors, self.n_predictors, &self.cross_predictors);
        let cholesky = gram.cholesky()?;
        let factor = cholesky.l();
        let mut minimum_diagonal = f64::INFINITY;
        let mut maximum_diagonal = 0.0_f64;
        for index in 0..self.n_predictors {
            let value = factor[(index, index)].abs();
            minimum_diagonal = minimum_diagonal.min(value);
            maximum_diagonal = maximum_diagonal.max(value);
        }
        if maximum_diagonal == 0.0
            || minimum_diagonal <= maximum_diagonal * FAST_PATH_DIAGONAL_RATIO
        {
            return None;
        }

        let cross =
            DMatrix::from_row_slice(self.n_predictors, self.n_responses, &self.cross_response);
        let coefficients = cholesky.solve(&cross);
        let explained = coefficients
            .iter()
            .zip(cross.iter())
            .map(|(coefficient, cross_value)| coefficient * cross_value)
            .sum::<f64>();
        if !explained.is_finite() {
            return None;
        }

        let residual = self.response_square_sum - explained;
        let scale = self.response_square_sum.abs().max(explained.abs()).max(1.0);
        if !residual.is_finite() || residual <= FAST_PATH_CANCELLATION_RATIO * scale {
            return None;
        }
        Some(residual)
    }
}

#[derive(Clone, Debug)]
pub struct CostLinear {
    shape: SignalShape,
    regression: RegressionData,
}

impl CostLinear {
    pub fn fit(signal: SignalView<'_>) -> Result<Self, Error> {
        let shape = signal.shape();
        if shape.n_features < 2 {
            return Err(Error::InsufficientFeatures {
                model: "linear",
                minimum: 2,
                actual: shape.n_features,
            });
        }
        let n_predictors = shape.n_features - 1;
        let mut response = Vec::with_capacity(shape.n_samples);
        let mut design = Vec::with_capacity(shape.n_samples * n_predictors);
        for row in signal.values().chunks_exact(shape.n_features) {
            response.push(row[0]);
            design.extend_from_slice(&row[1..]);
        }
        Ok(Self {
            shape,
            regression: RegressionData::new(design, response, shape.n_samples, n_predictors, 1),
        })
    }

    /// Fit an explicit response/design pair without the Python-compatible
    /// packed first-column convention.
    pub fn fit_response_design(
        response: SignalView<'_>,
        design: SignalView<'_>,
    ) -> Result<Self, Error> {
        if response.shape().n_samples != design.shape().n_samples {
            return Err(Error::SampleCountMismatch {
                response: response.shape().n_samples,
                design: design.shape().n_samples,
            });
        }
        let shape = SignalShape {
            n_samples: response.shape().n_samples,
            n_features: response.shape().n_features + design.shape().n_features,
        };
        Ok(Self {
            shape,
            regression: RegressionData::new(
                design.values().to_vec(),
                response.values().to_vec(),
                shape.n_samples,
                design.shape().n_features,
                response.shape().n_features,
            ),
        })
    }
}

impl SegmentCost for CostLinear {
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
        self.regression.residual(segment)
    }

    fn costs_ending_at(
        &self,
        starts: &[usize],
        end: usize,
        output: &mut Vec<f64>,
    ) -> Result<(), Error> {
        self.regression.costs_ending_at(starts, end, 0, 2, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_batch_matches_scalar(cost: &CostLinear, starts: &[usize], end: usize) {
        let mut batch = Vec::new();
        cost.costs_ending_at(starts, end, &mut batch).unwrap();
        assert_eq!(batch.len(), starts.len());
        for (&start, &actual) in starts.iter().zip(&batch) {
            let expected = cost.cost(start..end).unwrap();
            let tolerance = 1.0e-9 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "segment=[{start}, {end}), actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn exact_linear_relation_has_zero_cost() {
        let values = [3., 1., 1., 5., 1., 2., 7., 1., 3., 9., 1., 4.];
        let cost = CostLinear::fit(SignalView::new(&values, 4, 3).unwrap()).unwrap();
        assert!(cost.cost(0..4).unwrap() < 1e-20);

        let response = [3., 5., 7., 9.];
        let design = [1., 1., 1., 2., 1., 3., 1., 4.];
        let explicit = CostLinear::fit_response_design(
            SignalView::new(&response, 4, 1).unwrap(),
            SignalView::new(&design, 4, 2).unwrap(),
        )
        .unwrap();
        assert!(explicit.cost(0..4).unwrap() < 1e-20);

        let collinear = [1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.];
        let singular = CostLinear::fit(SignalView::new(&collinear, 4, 3).unwrap()).unwrap();
        assert!(singular.cost(0..4).unwrap().is_finite());
    }

    #[test]
    fn endpoint_batch_matches_svd_for_regular_singular_and_large_offset_data() {
        let mut regular = Vec::new();
        for index in 0..30 {
            let x = index as f64 / 7.0;
            let noise = ((index * 11) % 7) as f64 * 0.013;
            regular.extend_from_slice(&[2.0 + 1.5 * x + noise, 1.0, x]);
        }
        let regular = CostLinear::fit(SignalView::new(&regular, 30, 3).unwrap()).unwrap();
        assert_batch_matches_scalar(&regular, &[0, 3, 7, 12, 20], 30);

        let mut singular = Vec::new();
        for index in 0..20 {
            let x = index as f64;
            singular.extend_from_slice(&[3.0 * x + 0.1, x, 2.0 * x]);
        }
        let singular = CostLinear::fit(SignalView::new(&singular, 20, 3).unwrap()).unwrap();
        assert_batch_matches_scalar(&singular, &[0, 2, 5, 10], 20);

        let mut offset = Vec::new();
        for index in 0..20 {
            let x = 1.0e12 + index as f64;
            offset.extend_from_slice(&[5.0 + 0.25 * x, 1.0, x]);
        }
        let offset = CostLinear::fit(SignalView::new(&offset, 20, 3).unwrap()).unwrap();
        assert_batch_matches_scalar(&offset, &[0, 2, 6, 12], 20);
    }
}
