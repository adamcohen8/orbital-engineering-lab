//! Public OGP-only extension sharing the authoritative propagation kernels.

#[path = "../../oel-orbit/src/ogp.rs"]
pub mod ogp;
#[cfg(feature = "python")]
#[path = "../../oel-orbit/src/numeric_bytes.rs"]
mod numeric_bytes;
#[cfg(feature = "python")]
#[path = "../../oel-orbit/src/ogp_python.rs"]
mod ogp_python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn oel_rust_ogp(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ogp_python::register(module)
}
