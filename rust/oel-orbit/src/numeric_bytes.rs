//! Immutable little-endian float64 transfer for optional Python kernels.
//!
//! Byte inputs are borrowed only on little-endian hosts with suitable
//! alignment. Python's immutable bytes retain storage for the whole call,
//! including a detached numerical section. Other buffers decode explicitly.

use std::borrow::Cow;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub(crate) fn decode_f64_bytes(values: &[u8]) -> PyResult<Cow<'_, [f64]>> {
    if values.len() % 8 != 0 {
        return Err(PyValueError::new_err(
            "numeric bytes must contain complete float64 values",
        ));
    }
    if cfg!(target_endian = "little")
        && values.as_ptr().align_offset(std::mem::align_of::<f64>()) == 0
    {
        // SAFETY: f64 admits all bit patterns; length and alignment are checked,
        // the immutable input outlives the borrowed result, and host byte order
        // matches the explicit little-endian wire format.
        let numbers =
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<f64>(), values.len() / 8) };
        return Ok(Cow::Borrowed(numbers));
    }
    Ok(Cow::Owned(
        values
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
    ))
}

pub(crate) fn output_bytes(py: Python<'_>, values: &[f64]) -> PyResult<Py<PyBytes>> {
    if cfg!(target_endian = "little") {
        // SAFETY: initialized f64 storage is readable as bytes and remains
        // alive while PyBytes copies it into its own immutable allocation.
        let source = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        return Ok(PyBytes::new(py, source).unbind());
    }
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    Ok(PyBytes::new(py, &bytes).unbind())
}

#[cfg(test)]
mod tests {
    use super::decode_f64_bytes;

    #[test]
    fn unaligned_wire_values_decode_without_changing_bits() {
        let values = [
            0.0_f64,
            -0.0,
            f64::INFINITY,
            f64::from_bits(0x7ff800000000002a),
        ];
        let mut bytes = vec![0_u8];
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let decoded = decode_f64_bytes(&bytes[1..]).unwrap();
        assert_eq!(
            decoded
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn complete_empty_wire_values_are_accepted() {
        assert!(decode_f64_bytes(&[]).unwrap().is_empty());
    }
}
