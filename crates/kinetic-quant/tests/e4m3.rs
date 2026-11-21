//! NVFP4 and F8_E4M3 against llama.cpp's Python reference, gguf-py's
//! `quants.py`, bit for bit. `data/e4m3.txt` holds what it gives for edge
//! blocks (zero, NaN-coded and largest scales, every code), random NVFP4
//! blocks and all 256 F8_E4M3 bytes.

use kinetic_quant::{dequantize, TensorType};

#[test]
fn values_match_gguf_py() {
    let mut seen = [0; 2];
    for line in include_str!("data/e4m3.txt").lines() {
        if line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let ty = match fields.next() {
            Some("NVFP4") => TensorType::NVFP4,
            Some("F8_E4M3") => TensorType::F8E4M3,
            other => panic!("unknown type {other:?}"),
        };
        let hex = fields.next().unwrap();
        let block: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let want: Vec<u32> = fields
            .map(|f| u32::from_str_radix(f, 16).unwrap())
            .collect();
        let mut got = vec![0.0; want.len()];
        dequantize(ty, &block, &mut got).unwrap();
        let got: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, want, "{ty} {hex}");
        seen[(ty == TensorType::F8E4M3) as usize] += 1;
    }
    assert_eq!(seen, [16, 16]);
}
