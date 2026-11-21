use kinetic_gguf::{Gguf, TensorInfo, TensorSource, Value};
use kinetic_quant::{dequantize, Matrix};

use crate::blocks::{Activation, Experts, SwiGlu};
use crate::LoadError;

/// Typed access to the tensors and hyperparameters of a model file.
pub(crate) struct Loader<'a> {
    file: &'a dyn TensorSource,
}

impl<'a> Loader<'a> {
    pub fn new(file: &'a dyn TensorSource) -> Self {
        Loader { file }
    }

    pub fn header(&self) -> &'a Gguf {
        self.file.header()
    }

    pub fn hparam(&self, key: &str) -> Result<&'a Value, LoadError> {
        self.header()
            .arch_key(key)
            .ok_or_else(|| LoadError(format!("missing hyperparameter {key}")))
    }

    pub fn usize(&self, key: &str) -> Result<usize, LoadError> {
        self.hparam(key)?
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| LoadError(format!("{key} is not an integer")))
    }

    pub fn f32(&self, key: &str) -> Result<f32, LoadError> {
        self.hparam(key)?
            .as_f32()
            .ok_or_else(|| LoadError(format!("{key} is not a float")))
    }

    /// [`Loader::usize`], or `default` when the file does not set `key`.
    pub fn usize_or(&self, key: &str, default: usize) -> Result<usize, LoadError> {
        match self.header().arch_key(key) {
            Some(_) => self.usize(key),
            None => Ok(default),
        }
    }

    /// [`Loader::f32`], or `default` when the file does not set `key`.
    pub fn f32_or(&self, key: &str, default: f32) -> Result<f32, LoadError> {
        match self.header().arch_key(key) {
            Some(_) => self.f32(key),
            None => Ok(default),
        }
    }

    /// Fails on the per-tensor scales llama.cpp multiplies a product, its
    /// input or the cache by, such as the second scale of NVFP4 weights:
    /// nothing here applies them, and the output would be wrong without.
    pub fn refuse_weight_scales(&self) -> Result<(), LoadError> {
        const SCALES: &[&str] = &[".scale", ".input_scale", ".k_scale", ".v_scale"];
        match self
            .header()
            .tensors
            .iter()
            .find(|t| SCALES.iter().any(|s| t.name.ends_with(s)))
        {
            Some(t) => Err(LoadError(format!(
                "tensor {} scales a weight, which is not supported yet",
                t.name
            ))),
            None => Ok(()),
        }
    }

    fn tensor(&self, name: &str) -> Result<&'a TensorInfo, LoadError> {
        self.header()
            .tensor(name)
            .ok_or_else(|| LoadError(format!("missing tensor {name}")))
    }

    /// A 2-D weight with `cols` inputs and `rows` outputs.
    pub fn matrix(&self, name: &str, cols: usize, rows: usize) -> Result<Matrix<'a>, LoadError> {
        let t = self.tensor(name)?;
        if t.dims != [cols as u64, rows as u64] {
            return Err(LoadError(format!(
                "{name} has shape {:?}, expected [{cols}, {rows}]",
                t.dims
            )));
        }
        Ok(Matrix::new(t.ty, self.file.tensor_data(t), rows, cols))
    }

    /// A 3-D weight of `count` matrices with `cols` inputs and `rows`
    /// outputs each.
    pub fn experts(
        &self,
        name: &str,
        cols: usize,
        rows: usize,
        count: usize,
    ) -> Result<Experts<'a>, LoadError> {
        let t = self.tensor(name)?;
        if t.dims != [cols as u64, rows as u64, count as u64] {
            return Err(LoadError(format!(
                "{name} has shape {:?}, expected [{cols}, {rows}, {count}]",
                t.dims
            )));
        }
        Ok(Experts {
            ty: t.ty,
            data: self.file.tensor_data(t),
            rows,
            cols,
            count,
        })
    }

    /// A small tensor dequantized to f32, checked to hold `len` values.
    pub fn vector(&self, name: &str, len: usize) -> Result<Vec<f32>, LoadError> {
        let t = self.tensor(name)?;
        if t.elements() != len as u64 {
            return Err(LoadError(format!(
                "{name} has {} values, expected {len}",
                t.elements()
            )));
        }
        let mut out = vec![0.0; len];
        dequantize(t.ty, self.file.tensor_data(t), &mut out)
            .map_err(|e| LoadError(format!("{name}: {e}")))?;
        Ok(out)
    }

    /// [`Loader::vector`] for a tensor the file may leave out.
    pub fn optional_vector(&self, name: &str, len: usize) -> Result<Option<Vec<f32>>, LoadError> {
        match self.header().tensor(name) {
            Some(_) => self.vector(name, len).map(Some),
            None => Ok(None),
        }
    }

    /// A per-layer array hyperparameter of exactly `len` values.
    pub fn array<T>(
        &self,
        key: &str,
        len: usize,
        item: impl Fn(&Value) -> Option<T>,
    ) -> Result<Vec<T>, LoadError> {
        let items = self
            .hparam(key)?
            .as_array()
            .filter(|a| a.len() == len)
            .ok_or_else(|| LoadError(format!("{key} is not an array of {len} values")))?;
        items
            .iter()
            .map(|v| item(v).ok_or_else(|| LoadError(format!("{key} has an invalid value"))))
            .collect()
    }

    /// A per-layer integer hyperparameter, stored either as an array of
    /// `len` values or as one value for every layer.
    pub fn usizes(&self, key: &str, len: usize) -> Result<Vec<usize>, LoadError> {
        match self.hparam(key)?.as_array() {
            Some(_) => self.array(key, len, |v| v.as_u64().map(|n| n as usize)),
            None => Ok(vec![self.usize(key)?; len]),
        }
    }

    /// The gated feed-forward of layer `i`, under the usual tensor names.
    pub fn swiglu(&self, i: usize, d: usize, n_ff: usize) -> Result<SwiGlu<'a>, LoadError> {
        Ok(SwiGlu {
            gate: self.matrix(&format!("blk.{i}.ffn_gate.weight"), d, n_ff)?,
            up: self.matrix(&format!("blk.{i}.ffn_up.weight"), d, n_ff)?,
            down: self.matrix(&format!("blk.{i}.ffn_down.weight"), n_ff, d)?,
            activation: Activation::Silu,
            folding: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use kinetic_gguf::TensorType;

    use super::*;

    struct Header(Gguf);

    impl TensorSource for Header {
        fn header(&self) -> &Gguf {
            &self.0
        }

        fn tensor_data(&self, _: &TensorInfo) -> &[u8] {
            &[]
        }
    }

    fn file(names: &[&str]) -> Header {
        Header(Gguf {
            version: 3,
            alignment: 32,
            data_offset: 0,
            metadata: Vec::new(),
            tensors: names
                .iter()
                .map(|&name| TensorInfo {
                    name: name.into(),
                    dims: vec![1],
                    ty: TensorType::F32,
                    offset: 0,
                    size: 4,
                })
                .collect(),
        })
    }

    #[test]
    fn weight_scales_are_refused() {
        let plain = file(&["blk.0.attn_q.weight", "blk.0.layer_output_scale.weight"]);
        assert!(Loader::new(&plain).refuse_weight_scales().is_ok());
        for name in [
            "blk.0.attn_q.scale",
            "blk.3.ffn_up_exps.input_scale",
            "blk.0.attn_k.k_scale",
            "output.scale",
        ] {
            let scaled = file(&["blk.0.attn_q.weight", name]);
            let err = Loader::new(&scaled).refuse_weight_scales().unwrap_err();
            assert!(err.0.contains(name), "{}", err.0);
        }
    }
}
