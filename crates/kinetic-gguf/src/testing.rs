//! Model files without weights, for tests: a header built in code, whose
//! tensors all read as zeros.

use std::cell::RefCell;

use memmap2::{Mmap, MmapMut};

use crate::{write_header, Gguf, TensorInfo, TensorOut, TensorSource, TensorType, Value};

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
    pub fn new(gguf: Gguf) -> Header {
        let largest = gguf.tensors.iter().map(|t| t.size).max().unwrap_or(0);
        let len = usize::try_from(largest).expect("tensor fits in memory");
        let zeros = MmapMut::map_anon(len.max(1))
            .and_then(MmapMut::make_read_only)
            .expect("reserving zeroed memory");
        Header {
            gguf,
            zeros,
            read: RefCell::new(Vec::new()),
        }
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
}
