//! Every family builds from the headers in `tests/headers`, with weights
//! that read as zeros, prints what it did not read, and builds what it
//! built before: `tests/headers/digests.txt` holds a hash of each built
//! structure (every tensor read with its shape and type, layer count,
//! state sizes) or the refusal.
//!
//! cargo test --release -p kinetic-models --test headers -- --nocapture
//!
//! After a change that is meant to alter what a family reads:
//!
//! KINETIC_BLESS=1 cargo test --release -p kinetic-models --test headers

#[path = "headers/check.rs"]
mod check;

use std::collections::{BTreeMap, BTreeSet};
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
    let mut digests = BTreeMap::new();
    for path in fixtures() {
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let b = check::build(&Header::read(&path).unwrap());
        println!("{name}.gguf, {}", b.report());
        digests.insert(name, (b.arch.clone(), b.digest()));
        if b.structure.is_ok() {
            built.insert(b.arch);
        }
    }
    compare_digests(&digests);
    let missing: Vec<_> = kinetic_models::architectures()
        .filter(|a| !built.contains(*a))
        .collect();
    assert!(
        missing.is_empty(),
        "no fixture builds {missing:?}; add one with the fixture example"
    );
}

/// Fails, naming each fixture and its architecture, when a family built
/// something other than what `digests.txt` records.
fn compare_digests(digests: &BTreeMap<String, (String, String)>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/headers/digests.txt");
    let now: String = digests
        .iter()
        .map(|(name, (_, digest))| format!("{name} {digest}\n"))
        .collect();
    if std::env::var_os("KINETIC_BLESS").is_some() {
        std::fs::write(&path, now).unwrap();
        return;
    }
    let before = std::fs::read_to_string(&path).unwrap_or_default();
    let before: BTreeMap<&str, &str> = before.lines().filter_map(|l| l.split_once(' ')).collect();
    let mut changed: Vec<String> = digests
        .iter()
        .filter(|(name, (_, digest))| before.get(name.as_str()) != Some(&digest.as_str()))
        .map(|(name, (arch, digest))| match before.get(name.as_str()) {
            Some(was) => format!("{name} ({arch}): {was} -> {digest}"),
            None => format!("{name} ({arch}): not in digests.txt"),
        })
        .collect();
    changed.extend(
        before
            .keys()
            .filter(|name| !digests.contains_key(**name))
            .map(|name| format!("{name}: in digests.txt but has no fixture")),
    );
    assert!(
        changed.is_empty(),
        "built structures changed:\n{}\nrun with KINETIC_BLESS=1 if that is intended",
        changed.join("\n")
    );
}
