//! Every family builds from the headers in `tests/headers`, with weights
//! that read as zeros, and prints what it did not read.
//!
//! cargo test --release -p kinetic-models --test headers -- --nocapture

#[path = "headers/check.rs"]
mod check;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use kinetic_gguf::testing::Header;

fn fixtures() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/headers");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    paths.sort();
    paths
}

#[test]
fn every_architecture_builds_from_its_header() {
    let mut built = BTreeSet::new();
    for path in fixtures() {
        let name = path.file_stem().unwrap().to_string_lossy();
        let b = check::build(&Header::read(&path).unwrap());
        println!("{name}.gguf, {}", b.report());
        if b.structure.is_ok() {
            built.insert(b.arch);
        }
    }
    let missing: Vec<_> = kinetic_models::architectures()
        .filter(|a| !built.contains(*a))
        .collect();
    assert!(
        missing.is_empty(),
        "no fixture builds {missing:?}; add one with the fixture example"
    );
}
