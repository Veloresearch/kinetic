//! Writes the header of a model file as a test fixture: every key and
//! tensor shape, no weights, long values such as the tokenizer's vocabulary
//! kept only as their length. Then builds the model from it and prints what
//! the family refused or did not read.
//!
//! cargo run --release -p kinetic-models --example fixture -- model.gguf [fixture.gguf] [--as llama]
//!
//! The input can be just the first bytes of a remote file, as long as they
//! hold its whole header: `curl -L -r 0-8388607 <url> -o head.gguf`.
//! `--as` reads the file as another architecture, to see how far an
//! unsupported one gets in an existing family.

#[path = "../tests/headers/check.rs"]
mod check;

use kinetic_gguf::testing::{strip, Header};
use kinetic_gguf::{Error, Gguf, TensorSource, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: fixture <model.gguf> [fixture.gguf] [--as <architecture>]";
    let (args, rename) = match args.iter().position(|a| a == "--as") {
        Some(i) => (&args[..i], Some(args.get(i + 1).ok_or(usage)?)),
        None => (&args[..], None),
    };
    let [input, out @ ..] = args else {
        return Err(usage.into());
    };
    let header = Header::read(input).map_err(|e| match e {
        Error::Truncated { .. } | Error::Invalid(_) => {
            format!("{input}: {e} (if this is the start of a file, take more of it)")
        }
        e => format!("{input}: {e}"),
    })?;
    let stripped = strip(header.header());
    if let Some(out) = out.first() {
        std::fs::write(out, &stripped)?;
        println!("{out}: {} bytes", stripped.len());
    }
    let mut gguf = Gguf::parse_header(&stripped)?;
    if let Some(to) = rename {
        let from = format!("{}.", gguf.architecture().ok_or("no architecture")?);
        for (key, value) in &mut gguf.metadata {
            if key == "general.architecture" {
                *value = Value::String(to.clone());
            } else if let Some(rest) = key.strip_prefix(&from) {
                *key = format!("{to}.{rest}");
            }
        }
    }
    println!("{}", check::build(&Header::new(gguf)).report());
    Ok(())
}
