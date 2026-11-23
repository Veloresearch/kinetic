//! Builds a model from a header without weights and says what it read.
//! Shared by the `headers` test and the `fixture` example.

// The example uses only part of this module.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::Path;

use kinetic_gguf::testing::Header;
use kinetic_gguf::TensorSource;
use kinetic_models::{Embedder, Model};

pub struct Built {
    pub arch: String,
    /// What the family built, one fact per line, or why it refused.
    pub structure: Result<String, String>,
    pub layers: Option<usize>,
    /// Tensors of each weight type in the file.
    pub types: BTreeMap<&'static str, usize>,
    pub unread_tensors: Vec<String>,
    /// `<arch>.` keys whose name appears nowhere in this crate's source.
    pub unread_keys: Vec<String>,
}

impl Built {
    /// A hash of the structure that stays the same across platforms and
    /// compiler versions (FNV-1a).
    pub fn digest(&self) -> String {
        match &self.structure {
            Ok(s) => {
                let hash = s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                    (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
                });
                format!("{hash:016x}")
            }
            Err(e) => format!("refused: {e}"),
        }
    }

    pub fn report(&self) -> String {
        let mut out = format!("{}: ", self.arch);
        match (&self.structure, self.layers) {
            (Ok(_), Some(n)) => write!(out, "{n} layers").unwrap(),
            (Ok(_), None) => out.push_str("built"),
            (Err(e), _) => write!(out, "refused: {e}").unwrap(),
        }
        let types: Vec<String> = self.types.iter().map(|(t, n)| format!("{t} {n}")).collect();
        write!(out, "\n  types: {}", types.join(", ")).unwrap();
        for (what, names) in [
            ("tensors not read", &self.unread_tensors),
            ("keys named nowhere", &self.unread_keys),
        ] {
            if !names.is_empty() {
                write!(out, "\n  {what} ({}): {}", names.len(), names.join(", ")).unwrap();
            }
        }
        out
    }
}

/// Loads `header` as Kinetic would load its file: embedding models through
/// [`Embedder`], everything else as a [`Model`].
pub fn build(header: &Header) -> Built {
    let g = header.header();
    let arch = g.architecture().unwrap_or("").to_owned();
    let embedder = kinetic_models::is_encoder(&arch) || g.arch_key("pooling_type").is_some();
    let built = match embedder {
        true => Embedder::load(header).map(|e| {
            let s = format!(
                "embedder {} pooling {:?} context {}\n",
                e.n_embd(),
                e.pooling(),
                e.n_ctx_train()
            );
            (s, None)
        }),
        false => kinetic_models::load(header).map(|m| (describe(&m), Some(m.n_layers()))),
    };
    let (structure, layers) = match built {
        Ok((s, layers)) => (Ok(s), layers),
        Err(e) => (Err(e), None),
    };
    let read = header.tensors_read();
    let structure = structure.map_err(|e| e.0).map(|mut s| {
        for name in &read {
            let t = g.tensor(name).expect("read tensors are in the header");
            writeln!(s, "{name} {:?} {}", t.dims, t.ty).unwrap();
        }
        s
    });
    let read: BTreeSet<&String> = read.iter().collect();
    let mut types = BTreeMap::new();
    for t in &g.tensors {
        *types.entry(t.ty.name()).or_default() += 1;
    }
    let source = source();
    let prefix = format!("{arch}.");
    Built {
        unread_tensors: g
            .tensors
            .iter()
            .filter(|t| !read.contains(&t.name))
            .map(|t| t.name.clone())
            .collect(),
        unread_keys: g
            .metadata
            .iter()
            .filter_map(|(k, _)| k.strip_prefix(&prefix))
            .filter(|k| !source.contains(&format!("\"{k}\"")))
            .map(str::to_owned)
            .collect(),
        arch,
        structure,
        layers,
        types,
    }
}

fn describe(m: &Model) -> String {
    let mut s = format!(
        "layers {} embd {} vocab {} mtp {}\n",
        m.n_layers(),
        m.n_embd(),
        m.n_vocab(),
        m.has_mtp()
    );
    // the state's size follows the head counts, head sizes, windows and
    // recurrent widths of every layer
    for len in [1, 4096, 1 << 20] {
        writeln!(s, "state {len} {}", m.state_bytes(len)).unwrap();
    }
    s
}

/// Every Rust file under this crate's `src`, joined.
fn source() -> String {
    fn walk(dir: &Path, out: &mut String) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push_str(&std::fs::read_to_string(&path).unwrap());
            }
        }
    }
    let mut out = String::new();
    walk(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut out);
    out
}
