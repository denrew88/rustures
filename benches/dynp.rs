use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant};

use _rustures::{CostL2, Dynp, SignalView};

fn measure(n_samples: usize, changes: usize) -> Duration {
    let values: Vec<f64> = (0..n_samples)
        .map(|index| {
            let level = (index / 50 % 5) as f64 * 4.0;
            level + ((index * 37 % 101) as f64 / 101.0 - 0.5)
        })
        .collect();
    let cost = CostL2::fit(SignalView::new(&values, n_samples, 1).unwrap()).unwrap();
    let detector = Dynp::new(1, 1).unwrap();

    let mut best = Duration::MAX;
    for _ in 0..3 {
        let started = Instant::now();
        let result = detector.predict_changes(&cost, changes).unwrap();
        black_box(result);
        best = best.min(started.elapsed());
    }
    best
}

fn main() {
    const CHANGES: usize = 5;
    println!("Dynp rolling-row benchmark ({CHANGES} changes, L2, jump=1)");
    for n_samples in [200, 400, 800] {
        let elapsed = measure(n_samples, CHANGES);
        println!(
            "n={n_samples}: {:.3} ms (best of 3)",
            elapsed.as_secs_f64() * 1_000.0
        );
    }

    let positions = 100_001;
    let score_rows = CHANGES + 2;
    let full_score_bytes = score_rows * positions * size_of::<f64>();
    let rolling_score_bytes = 2 * positions * size_of::<f64>();
    let predecessor_bytes = score_rows * positions * size_of::<usize>();
    println!("memory estimate at n=100000:");
    println!("full score table: {full_score_bytes} bytes");
    println!("rolling score rows: {rolling_score_bytes} bytes");
    println!("predecessor table: {predecessor_bytes} bytes");
}
