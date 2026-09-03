use super::*;
use crate::{oracle, Dynp, Pelt};

#[test]
fn every_stage7_cost_matches_exhaustive_dynp_and_pelt() {
    let scalar = [0.0, 0.2, -0.1, 0.1, 0.0, 5.0, 5.2, 4.9, 5.1, 5.0];
    let scalar_view = SignalView::new(&scalar, 10, 1).unwrap();
    let mut cases = vec![
        CostSpec::L1.fit(scalar_view).unwrap(),
        CostSpec::Rank.fit(scalar_view).unwrap(),
        CostSpec::Normal { ridge: 1.0e-6 }.fit(scalar_view).unwrap(),
        CostSpec::AR { order: 1 }.fit(scalar_view).unwrap(),
        CostSpec::CLinear.fit(scalar_view).unwrap(),
        CostSpec::Mahalanobis { metric: None }
            .fit(scalar_view)
            .unwrap(),
    ];
    let mut linear_values = Vec::new();
    for index in 0..10 {
        let x = index as f64;
        let response = if index < 5 { 1.0 + x } else { 15.0 - x };
        linear_values.extend_from_slice(&[response, 1.0, x]);
    }
    cases.push(
        CostSpec::Linear
            .fit(SignalView::new(&linear_values, 10, 3).unwrap())
            .unwrap(),
    );

    for cost in cases {
        let min_size = cost.min_size();
        let expected_fixed =
            oracle::best_fixed_changes(10, min_size, 1, |range| cost.cost(range).unwrap())
                .unwrap()
                .unwrap();
        let actual_fixed = Dynp::new(min_size, 1)
            .unwrap()
            .predict_changes(&cost, 1)
            .unwrap();
        assert_eq!(actual_fixed.breakpoints, expected_fixed.0);

        let penalty = 1.5;
        let expected_penalized =
            oracle::best_penalized(10, min_size, penalty, |range| cost.cost(range).unwrap())
                .unwrap();
        let actual_penalized = Pelt::new(min_size, 1)
            .unwrap()
            .predict_penalty(&cost, penalty)
            .unwrap();
        assert_eq!(actual_penalized.breakpoints, expected_penalized.0);
        assert!((actual_penalized.objective - expected_penalized.1).abs() < 1.0e-9);
    }
}
