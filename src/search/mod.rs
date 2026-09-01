mod dynp;
mod greedy;
mod kernel_cpd;
mod l1_potts;
mod pelt;

pub use dynp::Dynp;
pub use greedy::{Binseg, BottomUp, Window};
pub use kernel_cpd::FusedKernelCPD;
pub use l1_potts::L1Potts;
pub use pelt::Pelt;

pub(crate) fn candidate_positions(n_samples: usize, jump: usize) -> Vec<usize> {
    let mut positions = vec![0];
    let mut position = jump;
    while position < n_samples {
        positions.push(position);
        match position.checked_add(jump) {
            Some(next) => position = next,
            None => break,
        }
    }
    positions.push(n_samples);
    positions
}
