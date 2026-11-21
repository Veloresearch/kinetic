use std::fmt;

use kinetic_quant::TensorType;

/// Errors from the CUDA driver and from this crate's own checks.
#[derive(Clone, Debug)]
pub enum Error {
    /// The CUDA driver library could not be loaded.
    DriverNotFound(String),
    /// The driver lacks an entry point this crate needs.
    MissingSymbol(&'static str),
    /// A driver call failed.
    Cuda {
        call: &'static str,
        code: i32,
        /// The driver's name for `code`, such as `CUDA_ERROR_OUT_OF_MEMORY`.
        name: &'static str,
    },
    /// The crate was compiled without nvcc and carries no kernels.
    KernelsNotBuilt,
    /// The kernels were built without code for this compute capability.
    KernelsNotBuiltFor((u32, u32)),
    /// No GEMV kernel exists for this weight type.
    Unsupported(TensorType),
    /// Sizes that do not fit together.
    Shape(String),
}

/// `Result` with this crate's [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

// a CUDA layer for another vendor's GPU fails here too, so say who CUDA is for
const NVIDIA_ONLY: &str =
    "the CUDA backend needs an NVIDIA GPU; AMD and Intel GPUs run through Vulkan (--device vulkan)";

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DriverNotFound(why) => write!(f, "CUDA driver not found: {why}; {NVIDIA_ONLY}"),
            Error::MissingSymbol(symbol) => {
                write!(f, "CUDA driver has no {symbol}; update the driver")
            }
            Error::Cuda { call, code, name } => write!(f, "{call} failed: {name} ({code})"),
            Error::KernelsNotBuilt => write!(
                f,
                "CUDA kernels not built: nvcc was not found when kinetic-cuda was compiled; {NVIDIA_ONLY}"
            ),
            Error::KernelsNotBuiltFor((major, minor)) => write!(
                f,
                "CUDA kernels not built for compute capability {major}.{minor}; \
                 Pascal and Volta (6.0 to 7.0) need a build with CUDA 12, \
                 and KINETIC_CUDA_ARCHS, when set, must include the GPU"
            ),
            Error::Unsupported(ty) => write!(f, "no CUDA GEMV kernel for {ty}"),
            Error::Shape(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_driver_or_build_says_cuda_is_for_nvidia() {
        let missing = Error::DriverNotFound("libamdhip64.so.6: cannot open".into());
        for e in [missing, Error::KernelsNotBuilt] {
            let msg = e.to_string();
            assert!(
                msg.contains("needs an NVIDIA GPU") && msg.contains("--device vulkan"),
                "{msg}"
            );
        }
    }
}
