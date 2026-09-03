use super::*;

#[test]
fn uses_componentwise_medians() {
    let values = [0., 10., 2., 8., 100., 6.];
    let cost = CostL1::fit(SignalView::new(&values, 3, 2).unwrap());
    assert_eq!(cost.cost(0..3).unwrap(), 104.0);
}
