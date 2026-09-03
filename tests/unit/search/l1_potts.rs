use super::*;

fn direct_objective(
    values: &[f64],
    weights: &[f64],
    states: &[usize],
    levels: &[f64],
    penalty: f64,
) -> f64 {
    let data = values
        .iter()
        .zip(weights)
        .zip(states)
        .map(|((&y, &w), &state)| w * (levels[state] - y).abs())
        .sum::<f64>();
    let changes = states.windows(2).filter(|pair| pair[0] != pair[1]).count();
    data + penalty * changes as f64
}

#[test]
fn matches_exhaustive_state_enumeration_with_weights() {
    let values = [0.0, 3.0, 2.0, 8.0, 8.0];
    let weights = [1.0, 0.5, 2.0, 1.5, 1.0];
    let view = SignalView::new(&values, values.len(), 1).unwrap();
    let solver = L1Potts::fit(view, Some(&weights)).unwrap();
    let penalty = 2.25;
    let result = solver.predict_penalty(penalty).unwrap();
    let k = solver.levels.len();
    let total = k.pow(values.len() as u32);
    let mut best = f64::INFINITY;
    for code in 0..total {
        let mut code = code;
        let mut states = vec![0; values.len()];
        for state in &mut states {
            *state = code % k;
            code /= k;
        }
        best = best.min(direct_objective(
            &values,
            &weights,
            &states,
            &solver.levels,
            penalty,
        ));
    }
    assert!((result.objective - best).abs() < 1.0e-10);
}

#[test]
fn rejects_multivariate_signal_and_bad_weights() {
    let values = [0.0, 1.0, 2.0, 3.0];
    let multi = SignalView::new(&values, 2, 2).unwrap();
    assert!(matches!(
        L1Potts::fit(multi, None),
        Err(Error::ScalarSignalRequired { .. })
    ));
    let scalar = SignalView::new(&values, 4, 1).unwrap();
    assert!(matches!(
        L1Potts::fit(scalar, Some(&[1.0, -1.0, 1.0, 1.0])),
        Err(Error::InvalidWeight { .. })
    ));
}
