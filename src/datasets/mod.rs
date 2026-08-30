use std::f64::consts::TAU;

use crate::Error;

#[derive(Clone, Debug)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / ((1u64 << 53) as f64))
    }

    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (TAU * self.uniform()).cos()
    }
}

fn validate_parameters(
    n_samples: usize,
    n_features: usize,
    n_bkps: usize,
    noise_std: f64,
) -> Result<(), Error> {
    if n_samples == 0
        || n_features == 0
        || n_bkps >= n_samples
        || !noise_std.is_finite()
        || noise_std < 0.0
    {
        return Err(Error::InvalidDatasetParameters);
    }
    if n_samples < n_bkps + 1 {
        return Err(Error::InvalidDatasetParameters);
    }
    Ok(())
}

fn breakpoints(n_samples: usize, n_bkps: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let segments = n_bkps + 1;
    let slack = n_samples - segments;
    let weights: Vec<f64> = (0..segments).map(|_| -rng.uniform().ln()).collect();
    let total: f64 = weights.iter().sum();
    let mut extras: Vec<usize> = weights
        .iter()
        .map(|weight| (slack as f64 * weight / total).floor() as usize)
        .collect();
    let mut remainder = slack - extras.iter().sum::<usize>();
    let mut order: Vec<usize> = (0..segments).collect();
    order.sort_by(|&left, &right| {
        let left_fraction = slack as f64 * weights[left] / total - extras[left] as f64;
        let right_fraction = slack as f64 * weights[right] / total - extras[right] as f64;
        right_fraction
            .total_cmp(&left_fraction)
            .then(left.cmp(&right))
    });
    for &index in &order {
        if remainder == 0 {
            break;
        }
        extras[index] += 1;
        remainder -= 1;
    }
    let mut end = 0usize;
    extras
        .into_iter()
        .map(|extra| {
            end += 1 + extra;
            end
        })
        .collect()
}

fn fill_noise(values: &mut [f64], noise_std: f64, rng: &mut SplitMix64) {
    if noise_std > 0.0 {
        for value in values {
            *value += noise_std * rng.normal();
        }
    }
}

pub fn piecewise_constant(
    n_samples: usize,
    n_features: usize,
    n_bkps: usize,
    noise_std: f64,
    seed: u64,
) -> Result<(Vec<f64>, Vec<usize>), Error> {
    validate_parameters(n_samples, n_features, n_bkps, noise_std)?;
    let mut rng = SplitMix64::new(seed);
    let bkps = breakpoints(n_samples, n_bkps, &mut rng);
    let mut values = vec![0.0; n_samples * n_features];
    let mut start = 0;
    for &end in &bkps {
        let means: Vec<f64> = (0..n_features)
            .map(|_| 10.0 * rng.uniform() - 5.0)
            .collect();
        for row in start..end {
            values[row * n_features..(row + 1) * n_features].copy_from_slice(&means);
        }
        start = end;
    }
    fill_noise(&mut values, noise_std, &mut rng);
    Ok((values, bkps))
}

pub fn piecewise_linear(
    n_samples: usize,
    n_features: usize,
    n_bkps: usize,
    noise_std: f64,
    seed: u64,
) -> Result<(Vec<f64>, Vec<usize>), Error> {
    validate_parameters(n_samples, n_features, n_bkps, noise_std)?;
    let mut rng = SplitMix64::new(seed);
    let bkps = breakpoints(n_samples, n_bkps, &mut rng);
    let mut values = vec![0.0; n_samples * n_features];
    let mut start = 0;
    for &end in &bkps {
        let intercepts: Vec<f64> = (0..n_features).map(|_| 6.0 * rng.uniform() - 3.0).collect();
        let slopes: Vec<f64> = (0..n_features).map(|_| 0.2 * rng.uniform() - 0.1).collect();
        for row in start..end {
            for feature in 0..n_features {
                values[row * n_features + feature] =
                    intercepts[feature] + slopes[feature] * (row - start) as f64;
            }
        }
        start = end;
    }
    fill_noise(&mut values, noise_std, &mut rng);
    Ok((values, bkps))
}

pub fn piecewise_normal(
    n_samples: usize,
    n_features: usize,
    n_bkps: usize,
    noise_std: f64,
    seed: u64,
) -> Result<(Vec<f64>, Vec<usize>), Error> {
    validate_parameters(n_samples, n_features, n_bkps, noise_std)?;
    let mut rng = SplitMix64::new(seed);
    let bkps = breakpoints(n_samples, n_bkps, &mut rng);
    let mut values = vec![0.0; n_samples * n_features];
    let mut start = 0;
    for &end in &bkps {
        let means: Vec<f64> = (0..n_features).map(|_| 8.0 * rng.uniform() - 4.0).collect();
        let scales: Vec<f64> = (0..n_features).map(|_| 0.5 + 1.5 * rng.uniform()).collect();
        for row in start..end {
            for feature in 0..n_features {
                values[row * n_features + feature] =
                    means[feature] + scales[feature] * rng.normal();
            }
        }
        start = end;
    }
    fill_noise(&mut values, noise_std, &mut rng);
    Ok((values, bkps))
}

pub fn piecewise_wavy(
    n_samples: usize,
    n_features: usize,
    n_bkps: usize,
    noise_std: f64,
    seed: u64,
) -> Result<(Vec<f64>, Vec<usize>), Error> {
    validate_parameters(n_samples, n_features, n_bkps, noise_std)?;
    let mut rng = SplitMix64::new(seed);
    let bkps = breakpoints(n_samples, n_bkps, &mut rng);
    let mut values = vec![0.0; n_samples * n_features];
    let mut start = 0;
    for &end in &bkps {
        for feature in 0..n_features {
            let amplitude = 0.5 + 2.5 * rng.uniform();
            let cycles = 0.5 + 3.5 * rng.uniform();
            let phase = TAU * rng.uniform();
            for row in start..end {
                let local = (row - start) as f64 / (end - start).max(1) as f64;
                values[row * n_features + feature] =
                    amplitude * (TAU * cycles * local + phase).sin();
            }
        }
        start = end;
    }
    fill_noise(&mut values, noise_std, &mut rng);
    Ok((values, bkps))
}

#[cfg(test)]
mod tests {
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
}
