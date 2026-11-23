//! Model files without weights, for tests: a header built in code or kept
//! from a real file by [`strip`], whose tensors all read as zeros.

use std::cell::RefCell;
use std::fs::File;
use std::path::Path;

use memmap2::{Mmap, MmapMut};

use crate::{write_header, Error, Gguf, TensorInfo, TensorOut, TensorSource, TensorType, Value};

/// A header whose tensors read as zeros. It remembers which tensors were
/// asked for, so a test can tell what a loader left unread.
pub struct Header {
    gguf: Gguf,
    // anonymous pages are only backed by memory once written, and nothing
    // writes them
    zeros: Mmap,
    read: RefCell<Vec<String>>,
}

/// A header with `keys` and `tensors`, each given by name, dimensions
/// (innermost first) and type.
///
/// Panics if a tensor's first dimension is not a whole number of blocks.
pub fn header(keys: &[(&str, Value)], tensors: &[(&str, &[u64], TensorType)]) -> Header {
    let metadata: Vec<(String, Value)> = keys
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    let tensors: Vec<TensorOut> = tensors
        .iter()
        .map(|&(name, dims, ty)| TensorOut {
            name: name.to_owned(),
            dims: dims.to_vec(),
            ty,
        })
        .collect();
    let mut buf = Vec::new();
    write_header(&mut buf, &metadata, &tensors).expect("writing to memory cannot fail");
    Header::new(Gguf::parse_header(&buf).expect("a header written here parses"))
}

impl Header {
    /// Panics if the largest tensor does not fit in the address space.
    pub fn new(gguf: Gguf) -> Header {
        Header::zeroed(gguf).expect("reserving zeroed memory")
    }

    fn zeroed(gguf: Gguf) -> Result<Header, Error> {
        let largest = gguf.tensors.iter().map(|t| t.size).max().unwrap_or(0);
        let len = usize::try_from(largest)
            .map_err(|_| Error::Invalid(format!("a tensor of {largest} bytes")))?;
        let zeros = MmapMut::map_anon(len.max(1))?.make_read_only()?;
        Ok(Header {
            gguf,
            zeros,
            read: RefCell::new(Vec::new()),
        })
    }

    /// Reads a header stored alone, or the first bytes of a model file as
    /// long as they hold the whole header.
    pub fn read(path: impl AsRef<Path>) -> Result<Header, Error> {
        let file = File::open(path)?;
        // SAFETY: read-only, and dropped before this returns
        let map = unsafe { Mmap::map(&file)? };
        Header::zeroed(Gguf::parse_header(&map)?)
    }

    /// Names of the tensors whose data was asked for, in order.
    pub fn tensors_read(&self) -> Vec<String> {
        self.read.borrow().clone()
    }
}

impl TensorSource for Header {
    fn header(&self) -> &Gguf {
        &self.gguf
    }

    fn tensor_data(&self, tensor: &TensorInfo) -> &[u8] {
        self.read.borrow_mut().push(tensor.name.clone());
        &self.zeros[..tensor.size as usize]
    }
}

/// Values longer than this are stored as their length by [`strip`].
pub const LONGEST_KEPT: usize = 256;

/// `gguf` as a header file without weights: every key and every tensor's
/// name, dimensions and type. Arrays of more than [`LONGEST_KEPT`] items
/// and longer strings, such as a tokenizer's vocabulary, merges and chat
/// template, become their length as a `U64`.
pub fn strip(gguf: &Gguf) -> Vec<u8> {
    let metadata: Vec<(String, Value)> = gguf
        .metadata
        .iter()
        .map(|(k, v)| {
            let len = match v {
                Value::Array(items) => items.len(),
                Value::String(s) => s.len(),
                _ => 0,
            };
            let v = match len > LONGEST_KEPT {
                true => Value::U64(len as u64),
                false => v.clone(),
            };
            (k.clone(), v)
        })
        .collect();
    let tensors: Vec<TensorOut> = gguf
        .tensors
        .iter()
        .map(|t| TensorOut {
            name: t.name.clone(),
            dims: t.dims.clone(),
            ty: t.ty,
        })
        .collect();
    let mut buf = Vec::new();
    write_header(&mut buf, &metadata, &tensors).expect("writing to memory cannot fail");
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_headers_read_zeros_and_remember_reads() {
        let h = header(
            &[("general.architecture", Value::String("llama".into()))],
            &[
                ("a.weight", &[64, 2], TensorType::Q8_0),
                ("b.weight", &[3], TensorType::F32),
            ],
        );
        assert_eq!(h.header().architecture(), Some("llama"));
        let b = h.header().tensor("b.weight").unwrap();
        assert_eq!(h.tensor_data(b), [0; 12]);
        let a = h.header().tensor("a.weight").unwrap();
        assert_eq!(h.tensor_data(a).len(), 4 * 34);
        assert_eq!(h.tensors_read(), ["b.weight", "a.weight"]);
    }

    #[test]
    fn strip_keeps_keys_and_tensors_and_shortens_long_values() {
        let vocab: Vec<Value> = (0..300).map(|i| Value::String(i.to_string())).collect();
        let layers = vec![Value::U32(8); 40];
        let h = header(
            &[
                ("general.architecture", Value::String("llama".into())),
                (
                    "llama.attention.head_count_kv",
                    Value::Array(layers.clone()),
                ),
                ("tokenizer.ggml.tokens", Value::Array(vocab)),
                ("tokenizer.ggml.eos_token_id", Value::U32(2)),
                ("tokenizer.chat_template", Value::String("x".repeat(1000))),
            ],
            &[("token_embd.weight", &[256, 300], TensorType::Q4K)],
        );
        let stripped = Gguf::parse_header(&strip(h.header())).unwrap();
        let get = |k| stripped.get(k).unwrap().clone();
        assert_eq!(get("llama.attention.head_count_kv"), Value::Array(layers));
        assert_eq!(get("tokenizer.ggml.tokens"), Value::U64(300));
        assert_eq!(get("tokenizer.ggml.eos_token_id"), Value::U32(2));
        assert_eq!(get("tokenizer.chat_template"), Value::U64(1000));
        let t = stripped.tensor("token_embd.weight").unwrap();
        assert_eq!((&t.dims[..], t.ty), (&[256, 300][..], TensorType::Q4K));
    }

    #[test]
    fn reading_garbage_is_an_error() {
        let path = std::env::temp_dir().join(format!("kinetic-garbage-{}", std::process::id()));
        let mut bytes = b"GGUF\x03\x00\x00\x00".to_vec();
        bytes.extend([0xff; 64]);
        std::fs::write(&path, &bytes).unwrap();
        assert!(Header::read(&path).is_err());
        let huge = TensorOut {
            name: "a".into(),
            dims: vec![1 << 40, 1 << 20],
            ty: TensorType::F32,
        };
        let mut bytes = Vec::new();
        write_header(&mut bytes, &[], &[huge]).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        assert!(Header::read(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
