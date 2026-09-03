use super::*;

#[test]
fn generators_are_seeded_and_well_shaped() {
    for generator in [
        piecewise_constant,
        piecewise_linear,
        piecewise_normal,
        piecewise_wavy,
    ] {
        let first = generator(50, 3, 4, 0.2, 7).unwrap();
        let second = generator(50, 3, 4, 0.2, 7).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.0.len(), 150);
        assert_eq!(first.1.len(), 5);
        assert_eq!(first.1.last(), Some(&50));
    }
}
