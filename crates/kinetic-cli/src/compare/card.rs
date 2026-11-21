//! The result card: a markdown table and a one-line summary to share.

use std::fmt::Write;

use serde_json::{json, Value};

use super::llama::Llama;
use super::{Settings, Speeds};
use crate::library::Local;
use crate::system::Machine;

pub struct Card<'a> {
    pub model: &'a Local,
    pub settings: &'a Settings,
    pub machine: Machine,
    pub kinetic_device: String,
    pub on_gpu: bool,
    pub kinetic: Speeds,
    /// Whether Kinetic's drafted edit output equalled plain decoding, on
    /// the first edit and on the repeat.
    pub first_edit_identical: bool,
    pub edit_identical: bool,
    pub llama: Llama,
}

/// (label, Kinetic, llama.cpp) for each test.
type Rows = [(&'static str, Option<f64>, Option<f64>); 4];

impl Card<'_> {
    fn rows(&self) -> Rows {
        let (k, l) = (&self.kinetic, &self.llama.speeds);
        [
            ("prompt pp512", Some(k.prompt), Some(l.prompt)),
            ("generate tg128", Some(k.generate), Some(l.generate)),
            ("edit256-first", k.first_edit, l.first_edit),
            ("edit256-repeated", k.edit, l.edit),
        ]
    }

    fn quant(&self) -> (String, String) {
        match &self.model.header {
            Ok(h) => (
                h.quant.clone(),
                format!("{:.2} GiB", h.size as f64 / (1u64 << 30) as f64),
            ),
            Err(_) => ("?".into(), "?".into()),
        }
    }

    fn hardware(&self) -> String {
        let m = &self.machine;
        let on = if self.on_gpu {
            &self.kinetic_device
        } else {
            &m.cpu
        };
        if self.on_gpu {
            format!("{on}, {}, {}", m.cpu, m.os)
        } else {
            format!("{on}, {} threads, {}", self.settings.threads, m.os)
        }
    }

    pub fn text(&self) -> String {
        let (quant, size) = self.quant();
        let version = env!("CARGO_PKG_VERSION");
        let mut s = String::new();
        let _ = writeln!(s, "## Kinetic vs llama.cpp: {}", self.model.name);
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "| test | Kinetic {version} | llama.cpp {} | ratio |",
            self.llama.version
        );
        let _ = writeln!(s, "| --- | ---: | ---: | ---: |");
        for (label, k, l) in self.rows() {
            let _ = writeln!(
                s,
                "| {label} | {} | {} | {} |",
                speed(k),
                speed(l),
                ratio(k, l)
            );
        }
        let _ = writeln!(s);
        let _ = writeln!(s, "{}", self.hardware());
        let device = if self.on_gpu {
            "all layers on the GPU"
        } else {
            "CPU only"
        };
        let _ = writeln!(
            s,
            "Same file ({size}, {quant}), {device}, {} threads, flash attention on, best of {}, t/s.",
            self.settings.threads, self.settings.repetitions
        );
        let yes = |identical| if identical { "yes" } else { "NO" };
        let _ = writeln!(
            s,
            "edit256: greedy copy of a file with one rename, first on a fresh engine, then repeated on it; llama.cpp speculative mode {}, no prompt cache; Kinetic's output identical to plain decoding: first {}, repeated {}.",
            self.llama.edit_mode,
            yes(self.first_edit_identical),
            yes(self.edit_identical)
        );
        let _ = writeln!(s);
        let _ = writeln!(s, "{}", self.summary());
        s
    }

    /// One line, for a post.
    fn summary(&self) -> String {
        let on = if self.on_gpu {
            self.kinetic_device.as_str()
        } else {
            self.machine.cpu.as_str()
        };
        let parts: Vec<String> = self
            .rows()
            .iter()
            .filter_map(|&(label, k, l)| {
                let name = label.split(' ').next().unwrap_or(label);
                Some(format!(
                    "{name} {:.0} vs {:.0} t/s ({})",
                    k?,
                    l?,
                    ratio(Some(k?), Some(l?))
                ))
            })
            .collect();
        format!(
            "{} on {on}: Kinetic {} vs llama.cpp {}: {}.",
            self.model.name,
            env!("CARGO_PKG_VERSION"),
            self.llama.version,
            parts.join(", ")
        )
    }

    pub fn json(&self) -> Value {
        let (quant, _) = self.quant();
        let tests: Vec<Value> = self
            .rows()
            .iter()
            .map(|(label, k, l)| {
                json!({
                    "test": label.rsplit(' ').next(),
                    "kinetic_tokens_per_second": k,
                    "llama_cpp_tokens_per_second": l,
                    "ratio": k.zip(*l).map(|(k, l)| k / l),
                })
            })
            .collect();
        json!({
            "model": self.model.name,
            "path": self.model.path.display().to_string(),
            "quantization": quant,
            "size_bytes": self.model.header.as_ref().ok().map(|h| h.size),
            "os": self.machine.os,
            "cpu": self.machine.cpu,
            "device": if self.on_gpu { self.kinetic_device.as_str() } else { "cpu" },
            "threads": self.settings.threads,
            "repetitions": self.settings.repetitions,
            "kinetic_version": env!("CARGO_PKG_VERSION"),
            "llama_cpp_version": self.llama.version,
            "llama_cpp_backend": self.llama.backend,
            "llama_cpp_gpu": self.llama.gpu,
            "llama_cpp_edit_mode": self.llama.edit_mode,
            "first_edit_identical": self.first_edit_identical,
            "edit_identical": self.edit_identical,
            "tests": tests,
        })
    }
}

fn speed(v: Option<f64>) -> String {
    v.map_or_else(|| "-".into(), |v| format!("{v:.1}"))
}

fn ratio(k: Option<f64>, l: Option<f64>) -> String {
    match (k, l) {
        (Some(k), Some(l)) if l > 0.0 => format!("{:.2}x", k / l),
        _ => "-".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::Store;
    use crate::Device;

    #[test]
    fn first_and_repeated_edits_are_separate_rows() {
        let model = Local {
            name: "m".into(),
            path: "m.gguf".into(),
            projector: None,
            store: Store::Folder,
            header: Err("not read".into()),
        };
        let settings = Settings {
            device: Device::Cpu,
            threads: 4,
            repetitions: 2,
        };
        let card = Card {
            model: &model,
            settings: &settings,
            machine: Machine {
                os: "os".into(),
                cpu: "cpu".into(),
                threads: 4,
                isa: Vec::new(),
                ram_total: 0,
                ram_available: 0,
            },
            kinetic_device: "the CPU".into(),
            on_gpu: false,
            kinetic: Speeds {
                prompt: 100.0,
                generate: 20.0,
                first_edit: Some(60.0),
                edit: Some(120.0),
            },
            first_edit_identical: true,
            edit_identical: false,
            llama: Llama {
                speeds: Speeds {
                    prompt: 100.0,
                    generate: 20.0,
                    first_edit: Some(30.0),
                    edit: Some(30.0),
                },
                version: "b1".into(),
                backend: "CPU".into(),
                gpu: String::new(),
                edit_mode: "ngram-simple".into(),
            },
        };
        let text = card.text();
        assert!(
            text.contains("| edit256-first | 60.0 | 30.0 | 2.00x |"),
            "{text}"
        );
        assert!(
            text.contains("| edit256-repeated | 120.0 | 30.0 | 4.00x |"),
            "{text}"
        );
        assert!(text.contains("first yes, repeated NO"), "{text}");
        let json = card.json();
        assert_eq!(json["tests"][2]["test"], "edit256-first");
        assert_eq!(json["tests"][3]["test"], "edit256-repeated");
        assert_eq!(json["first_edit_identical"], true);
    }
}
