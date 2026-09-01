use std::ops::Range;

use _rustures::{CostAR, CostLinear, Dynp, Error, Pelt, SegmentCost, SignalView};

#[derive(Clone)]
struct ScalarOnly<C>(C);

impl<C: SegmentCost> SegmentCost for ScalarOnly<C> {
    fn n_samples(&self) -> usize {
        self.0.n_samples()
    }

    fn n_features(&self) -> usize {
        self.0.n_features()
    }

    fn min_size(&self) -> usize {
        self.0.min_size()
    }

    fn cost(&self, segment: Range<usize>) -> Result<f64, Error> {
        self.0.cost(segment)
    }
}

fn deterministic_noise(seed: u64, row: usize, column: usize) -> f64 {
    let mut value = seed
        .wrapping_add((row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((column as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value as f64 / u64::MAX as f64) - 0.5
}

fn linear_signal(
    n_samples: usize,
    n_predictors: usize,
    offset: f64,
    collinear: bool,
    seed: u64,
) -> Vec<f64> {
    let mut packed = Vec::with_capacity(n_samples * (n_predictors + 1));
    for row in 0..n_samples {
        let x = row as f64 / n_samples as f64;
        let segment = (3 * row / n_samples).min(2) as f64;
        let mut predictors = Vec::with_capacity(n_predictors);
        for column in 0..n_predictors {
            let value = if column == 0 {
                1.0
            } else if collinear && column + 1 == n_predictors {
                2.0 * predictors[(column - 1).max(0)]
            } else {
                ((column + 1) as f64 * x).sin() + x.powi((column % 3 + 1) as i32)
            };
            predictors.push(value);
        }
        let response = offset
            + predictors
                .iter()
                .enumerate()
                .map(|(column, value)| value * (segment + 1.0) * (column as f64 + 0.5))
                .sum::<f64>()
            + 0.03 * deterministic_noise(seed, row, 0);
        packed.push(response);
        packed.extend_from_slice(&predictors);
    }
    packed
}

fn ar_signal(
    n_samples: usize,
    n_features: usize,
    offset: f64,
    correlated: bool,
    seed: u64,
) -> Vec<f64> {
    let mut values = Vec::with_capacity(n_samples * n_features);
    for row in 0..n_samples {
        let segment = (3 * row / n_samples).min(2) as f64;
        let time = row as f64;
        let base =
            offset + 2.5 * segment + (0.11 * time).sin() + 0.08 * deterministic_noise(seed, row, 0);
        for feature in 0..n_features {
            let value = if correlated && feature > 0 {
                (feature as f64 + 1.0) * base
            } else {
                base + 0.4 * feature as f64 * (0.07 * time * (feature + 1) as f64).cos()
                    + 0.05 * deterministic_noise(seed, row, feature + 1)
            };
            values.push(value);
        }
    }
    values
}

fn assert_dynp_batch_matches_scalar<C: SegmentCost + Clone>(
    cost: C,
    changes: usize,
    jump: usize,
    label: &str,
) {
    let detector = Dynp::new(cost.min_size(), jump).unwrap();
    let batch = detector.predict_changes(&cost, changes).unwrap();
    let scalar = detector
        .predict_changes(&ScalarOnly(cost), changes)
        .unwrap();
    assert_eq!(batch.breakpoints, scalar.breakpoints, "{label}");
    assert!(
        regression_objectives_close(batch.segment_cost, scalar.segment_cost),
        "{label}: batch={}, scalar={}",
        batch.segment_cost,
        scalar.segment_cost
    );
}

fn assert_pelt_batch_matches_scalar<C: SegmentCost + Clone>(
    cost: C,
    penalty: f64,
    jump: usize,
    label: &str,
) {
    let detector = Pelt::new(cost.min_size(), jump).unwrap();
    let batch = detector.predict_penalty(&cost, penalty).unwrap();
    let scalar = detector
        .predict_penalty(&ScalarOnly(cost), penalty)
        .unwrap();
    assert_eq!(batch.breakpoints, scalar.breakpoints, "{label}");
    assert!(
        regression_objectives_close(batch.objective, scalar.objective),
        "{label}: batch={}, scalar={}",
        batch.objective,
        scalar.objective
    );
}

fn regression_objectives_close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-6 * left.abs().max(right.abs()).max(1.0)
}

#[test]
#[ignore = "release-mode Linear/AR mass validation"]
fn regression_endpoint_batches_match_scalar_svd_massively() {
    let mut dynp_cases = 0usize;
    let mut pelt_cases = 0usize;

    for seed in [7_u64, 20260901] {
        for n_samples in [18, 31, 47] {
            for n_predictors in [1, 2, 4, 8] {
                for offset in [0.0, 1.0e6, 1.0e12, 1.0e15] {
                    for collinear in [false, true] {
                        let values =
                            linear_signal(n_samples, n_predictors, offset, collinear, seed);
                        let cost = CostLinear::fit(
                            SignalView::new(&values, n_samples, n_predictors + 1).unwrap(),
                        )
                        .unwrap();
                        for jump in [1, 3] {
                            for changes in [1, 2] {
                                let label = format!(
                                    "linear seed={seed} n={n_samples} q={n_predictors} offset={offset} collinear={collinear} jump={jump} K={changes}"
                                );
                                assert_dynp_batch_matches_scalar(
                                    cost.clone(),
                                    changes,
                                    jump,
                                    &label,
                                );
                                dynp_cases += 1;
                            }
                        }
                        if seed == 7 && n_samples == 31 {
                            let label = format!(
                                "linear-pelt q={n_predictors} offset={offset} collinear={collinear}"
                            );
                            assert_pelt_batch_matches_scalar(cost, 0.5, 3, &label);
                            pelt_cases += 1;
                        }
                    }
                }
            }
        }
    }

    for seed in [11_u64, 20260901] {
        for n_samples in [20, 32, 44] {
            for n_features in [1, 2, 4] {
                for order in [1, 2, 4, 8] {
                    for offset in [0.0, 1.0e6, 1.0e12] {
                        for correlated in [false, true] {
                            let values = ar_signal(n_samples, n_features, offset, correlated, seed);
                            let cost = CostAR::fit(
                                SignalView::new(&values, n_samples, n_features).unwrap(),
                                order,
                            )
                            .unwrap();
                            for jump in [1, 3] {
                                for changes in [1, 2] {
                                    if (changes + 1) * cost.min_size() > n_samples {
                                        continue;
                                    }
                                    let label = format!(
                                        "ar seed={seed} n={n_samples} d={n_features} order={order} offset={offset} correlated={correlated} jump={jump} K={changes}"
                                    );
                                    assert_dynp_batch_matches_scalar(
                                        cost.clone(),
                                        changes,
                                        jump,
                                        &label,
                                    );
                                    dynp_cases += 1;
                                }
                            }
                            if seed == 11 && n_samples == 32 {
                                let label = format!(
                                    "ar-pelt d={n_features} order={order} offset={offset} correlated={correlated}"
                                );
                                assert_pelt_batch_matches_scalar(cost, 1.0, 3, &label);
                                pelt_cases += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    println!("validated Dynp cases: {dynp_cases}");
    println!("validated Pelt cases: {pelt_cases}");
    assert!(dynp_cases >= 2_000);
    assert!(pelt_cases >= 50);
}
