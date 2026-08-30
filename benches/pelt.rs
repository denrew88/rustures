use std::hint::black_box;
use std::time::{Duration, Instant};

use _rustures::{CostL2, Pelt, SignalView};

#[derive(Clone, Copy)]
enum Scenario {
    NoChange,
    Sparse,
    Frequent,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::NoChange => "no-change",
            Self::Sparse => "sparse",
            Self::Frequent => "frequent",
        }
    }
}

fn signal(n_samples: usize, scenario: Scenario) -> Vec<f64> {
    (0..n_samples)
        .map(|index| {
            let level = match scenario {
                Scenario::NoChange => 0.0,
                Scenario::Sparse => {
                    [0.0, 7.0, -4.0, 5.0][(index.saturating_mul(4) / n_samples).min(3)]
                }
                Scenario::Frequent => [0.0, 7.0, -4.0, 5.0][(index / 50) % 4],
            };
            let noise_code = (index * 48_271 + index * index * 31) % 104_729;
            level + (noise_code as f64 / 104_729.0 - 0.5) * 0.1
        })
        .collect()
}

fn measure(n_samples: usize, scenario: Scenario, penalty: f64) -> (Duration, usize) {
    let values = signal(n_samples, scenario);
    let cost = CostL2::fit(SignalView::new(&values, n_samples, 1).unwrap()).unwrap();
    let detector = Pelt::new(2, 1).unwrap();

    let mut best = Duration::MAX;
    let mut changes = 0;
    for _ in 0..3 {
        let started = Instant::now();
        let result = detector.predict_penalty(&cost, penalty).unwrap();
        best = best.min(started.elapsed());
        changes = result.breakpoints.len() - 1;
        black_box(result);
    }
    (best, changes)
}

fn main() {
    const PENALTY: f64 = 5.0;
    println!("PELT benchmark (L2, min_size=2, jump=1, penalty={PENALTY})");
    for scenario in [Scenario::NoChange, Scenario::Sparse, Scenario::Frequent] {
        for n_samples in [500, 1_000, 2_000, 4_000] {
            let (elapsed, changes) = measure(n_samples, scenario, PENALTY);
            println!(
                "scenario={}, n={n_samples}, changes={changes}: {:.3} ms (best of 3)",
                scenario.name(),
                elapsed.as_secs_f64() * 1_000.0
            );
        }
    }
}
