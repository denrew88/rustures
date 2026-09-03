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
