//! Attention over depth (Inkling 2): in place of one residual stream, a
//! layer reads a mix of the outputs of earlier blocks, weighted by a
//! softmax of learned queries against them; a looped layer also takes in a
//! message from the previous pass through a low-rank gate.

use kinetic_cpu::{dot, matmul, pool, rms_norm};
use kinetic_quant::Matrix;

/// Epsilon of the norms inside these blocks, fixed by the reference.
const EPS: f32 = 1e-6;

/// For each query of `queries`, `d` values each: per token, the states of
/// `states` normalized by their root mean square, weighted by the softmax
/// of their dot products with the query over the square root of `d`.
/// Every state holds the same number of rows of `d` values.
pub fn mix(states: &[&[f32]], queries: &[f32], d: usize) -> Vec<Vec<f32>> {
    let n = states[0].len() / d;
    let n_query = queries.len() / d;
    let ones = vec![1.0; d];
    let scale = 1.0 / (d as f32).sqrt();
    let rows = pool::map(n, |t| {
        let normed: Vec<Vec<f32>> = states
            .iter()
            .map(|s| {
                let mut row = s[t * d..][..d].to_vec();
                rms_norm(&mut row, &ones, EPS);
                row
            })
            .collect();
        let mut out = vec![0.0; n_query * d];
        for (q, out) in queries.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
            let mut p: Vec<f32> = normed.iter().map(|s| dot(s, q) * scale).collect();
            kinetic_cpu::softmax(&mut p);
            for (s, p) in normed.iter().zip(p) {
                out.iter_mut().zip(s).for_each(|(o, s)| *o += p * s);
            }
        }
        out
    });
    (0..n_query)
        .map(|q| {
            rows.iter()
                .flat_map(|r| &r[q * d..][..d])
                .copied()
                .collect()
        })
        .collect()
}

/// Mixes a message from the previous pass into the input of a looped layer:
/// `x - forget x + gain message`, where `forget` and `gain` are `tanh` of
/// projections of one low-rank code of the normalized input and message.
pub struct Control<'a> {
    /// The input and message, side by side, to the code.
    pub down: Matrix<'a>,
    pub gain: Matrix<'a>,
    pub forget: Matrix<'a>,
}

impl Control<'_> {
    /// Applies the control to every row of `x` with its row of `message`.
    pub fn apply(&self, x: &mut [f32], message: &[f32]) {
        let d = self.gain.rows;
        let n = x.len() / d;
        let ones = vec![1.0; d];
        let mut joined = Vec::with_capacity(2 * x.len());
        for (x, m) in x.chunks_exact(d).zip(message.chunks_exact(d)) {
            for part in [x, m] {
                let mut row = part.to_vec();
                rms_norm(&mut row, &ones, EPS);
                joined.extend(row);
            }
        }
        let mut code = vec![0.0; n * self.down.rows];
        matmul(&self.down, &joined, &mut code);
        let mut gain = vec![0.0; n * d];
        let mut forget = vec![0.0; n * d];
        matmul(&self.gain, &code, &mut gain);
        matmul(&self.forget, &code, &mut forget);
        for (i, x) in x.iter_mut().enumerate() {
            *x = (*x - forget[i].tanh() * *x) + gain[i].tanh() * message[i];
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use kinetic_quant::TensorType;

    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-5, "value {i}: {g} against {w}");
        }
    }

    pub(crate) fn bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// `n` values `((i * mul) % m - m / 2) * scale`, the same in the NumPy
    /// scripts the expected values come from.
    pub(crate) fn pattern(n: usize, mul: usize, m: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * mul % m) as f32 - (m / 2) as f32) * scale)
            .collect()
    }

    pub(crate) fn f32_matrix(bytes: &[u8], rows: usize, cols: usize) -> Matrix<'_> {
        Matrix::new(TensorType::F32, bytes, rows, cols)
    }

    // Expected values from scripts in NumPy (float64) of the reference's
    // graph: `mix` and `control` in llama.cpp's src/models/inkling2.cpp.
    #[test]
    fn mix_weights_normalized_states_by_query() {
        let a = [1.0, -2.0, 0.5, 3.0, 0.0, 1.0, 2.0, -1.0];
        let b = [0.5, 0.5, -1.0, 2.0, 1.5, -0.5, 0.0, 1.0];
        let c = [-1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let queries = [0.3, -0.7, 1.1, 0.2, -0.4, 0.9, 0.0, 0.5];
        let out = mix(&[&a, &b, &c], &queries, 4);
        close(&out[0], &MIX_Q0);
        close(&out[1], &MIX_Q1);
    }

    #[test]
    fn control_forgets_and_gains_the_message() {
        let mut x = [0.4, -1.2, 2.0, 0.7, 1.0, 0.0, -0.5, 0.25];
        let message = [1.0, 0.5, -0.5, 2.0, -1.0, 1.5, 0.5, 0.0];
        let down = bytes(&pattern(2 * 8, 7, 11, 0.1));
        let gain = bytes(&pattern(4 * 2, 5, 7, 0.3));
        let forget = bytes(&pattern(4 * 2, 3, 5, 0.4));
        let control = Control {
            down: f32_matrix(&down, 2, 8),
            gain: f32_matrix(&gain, 4, 2),
            forget: f32_matrix(&forget, 4, 2),
        };
        control.apply(&mut x, &message);
        close(&x, &CONTROL);
    }

    const MIX_Q0: [f32; 8] = [
        -0.09807524,
        -0.3801859,
        0.7416793,
        1.252107,
        0.8653104,
        0.4108964,
        0.8732802,
        0.4029266,
    ];
    const MIX_Q1: [f32; 8] = [
        -0.003087627,
        -0.03862286,
        0.2827281,
        1.358669,
        0.7962813,
        0.6192766,
        0.9999977,
        0.4155602,
    ];
    const CONTROL: [f32; 8] = [
        0.3916608, -1.442781, 1.174851, 2.096837, 0.6495011, 0.8654389, -0.5473431, 0.2258512,
    ];
}
