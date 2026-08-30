use std::hint::black_box;
use std::time::{Duration, Instant};

use _rustures::{CostL2, SegmentCost, SignalView};

fn measure(cost: &CostL2, segment_len: usize, queries: usize) -> Duration {
    let available_starts = cost.n_samples() - segment_len + 1;
    let started = Instant::now();
    for query in 0..queries {
        let start = black_box(query % available_starts);
        black_box(cost.cost(start..start + segment_len).unwrap());
    }
    started.elapsed()
}

fn main() {
    const N_SAMPLES: usize = 200_000;
    const N_FEATURES: usize = 8;
    const QUERIES: usize = 500_000;

    let values: Vec<f64> = (0..N_SAMPLES * N_FEATURES)
        .map(|index| ((index * 17 % 1009) as f64).sin())
        .collect();
    let cost = CostL2::fit(SignalView::new(&values, N_SAMPLES, N_FEATURES).unwrap()).unwrap();

    let short = measure(&cost, 16, QUERIES);
    let long = measure(&cost, N_SAMPLES / 2, QUERIES);
    let short_ns = short.as_nanos() as f64 / QUERIES as f64;
    let long_ns = long.as_nanos() as f64 / QUERIES as f64;

    println!("CostL2 query benchmark ({N_FEATURES} features, {QUERIES} queries)");
    println!("length=16: {short_ns:.2} ns/query");
    println!("length={}: {long_ns:.2} ns/query", N_SAMPLES / 2);
    println!("long/short ratio: {:.3}", long_ns / short_ns);
}
