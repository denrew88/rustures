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
