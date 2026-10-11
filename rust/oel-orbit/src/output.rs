//! Bounded JSON text traversal with Python's scalar spelling contract.
//!
//! The Python owner still sanitizes uncommon scalar/container types and owns
//! file paths, atomic replacement and failures. Native traversal only avoids
//! repeatedly yielding each primitive/punctuation token to Python.

use std::collections::HashSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyDict, PyFloat, PyInt, PyIterator, PyList, PyModule, PyString, PyTuple,
};

const CHUNK_BYTES: usize = 2 * 1024 * 1024;

enum Frame {
    Value(Py<PyAny>, usize),
    Sequence(Py<PyAny>, usize, usize, usize),
    Mapping(Py<PyDict>, usize, usize, usize, Py<PyIterator>),
    Fallback(Py<PyIterator>),
    Text(String, usize),
    Close(usize, usize, char),
}

#[pyclass(module = "oel_rust_orbit")]
pub struct JSONSafeChunks {
    frames: Vec<Frame>,
    active: HashSet<usize>,
    fallback: Py<PyAny>,
    quote: Py<PyAny>,
}

impl JSONSafeChunks {
    fn newline(level: usize) -> String {
        format!("\n{}", "  ".repeat(level))
    }

    fn push_value(&mut self, py: Python<'_>, value: Py<PyAny>, level: usize) -> PyResult<()> {
        let item = value.bind(py);
        let scalar = if item.is_none() {
            Some("null".to_owned())
        } else if item.is_exact_instance_of::<PyBool>() {
            Some(if item.is_truthy()? { "true" } else { "false" }.to_owned())
        } else if item.is_exact_instance_of::<PyString>() {
            Some(self.quote.bind(py).call1((item,))?.extract::<String>()?)
        } else if item.is_exact_instance_of::<PyInt>() {
            Some(item.repr()?.extract::<String>()?)
        } else if item.is_exact_instance_of::<PyFloat>() {
            Some(if item.extract::<f64>()?.is_finite() {
                item.repr()?.extract::<String>()?
            } else {
                "null".to_owned()
            })
        } else {
            None
        };
        if let Some(text) = scalar {
            self.frames.push(Frame::Text(text, 0));
            return Ok(());
        }
        let marker = item.as_ptr() as usize;
        if item.is_exact_instance_of::<PyList>() || item.is_exact_instance_of::<PyTuple>() {
            let len = item.len()?;
            if len == 0 {
                self.frames.push(Frame::Text("[]".to_owned(), 0));
                return Ok(());
            }
            if !self.active.insert(marker) {
                return Err(PyValueError::new_err("Circular reference detected"));
            }
            self.frames.push(Frame::Close(marker, level, ']'));
            self.frames.push(Frame::Sequence(value, 0, len, level + 1));
            self.frames
                .push(Frame::Text(format!("[{}", Self::newline(level + 1)), 0));
        } else if item.is_exact_instance_of::<PyDict>() {
            let original = item.cast::<PyDict>()?;
            // PyDict's Rust iterator panics on mutation. Exact built-in scalar
            // key spelling cannot invoke user code; uncommon key __str__ can,
            // so retain Python iteration and its RuntimeError/cleanup behavior.
            if original.iter().any(|(key, _)| {
                !(key.is_none()
                    || key.is_exact_instance_of::<PyString>()
                    || key.is_exact_instance_of::<PyBool>()
                    || key.is_exact_instance_of::<PyInt>()
                    || key.is_exact_instance_of::<PyFloat>())
            }) {
                let iterator = self
                    .fallback
                    .bind(py)
                    .call1((item, level))?
                    .try_iter()?
                    .unbind();
                self.frames.push(Frame::Fallback(iterator));
                return Ok(());
            }
            let normalized = PyDict::new(py);
            // Match str-key conversion, first insertion order, last value wins.
            for (key, child) in original.iter() {
                normalized.set_item(key.str()?, child)?;
            }
            let len = normalized.len();
            if len == 0 {
                self.frames.push(Frame::Text("{}".to_owned(), 0));
                return Ok(());
            }
            if !self.active.insert(marker) {
                return Err(PyValueError::new_err("Circular reference detected"));
            }
            let keys = normalized.keys().try_iter()?.unbind();
            self.frames.push(Frame::Close(marker, level, '}'));
            self.frames
                .push(Frame::Mapping(normalized.unbind(), 0, len, level + 1, keys));
            self.frames
                .push(Frame::Text(format!("{{{}", Self::newline(level + 1)), 0));
        } else {
            // NumPy/Path and custom subclasses retain the established Python
            // sanitizer and error behavior rather than adopting new coercions.
            let iterator = self
                .fallback
                .bind(py)
                .call1((item, level))?
                .try_iter()?
                .unbind();
            self.frames.push(Frame::Fallback(iterator));
        }
        Ok(())
    }
}

#[pymethods]
impl JSONSafeChunks {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<String>> {
        let result = self.next_chunk(py);
        if result.is_err() {
            // Like a Python generator, a failed traversal is exhausted. Do
            // not retain partial container frames or allow malformed restart.
            self.frames.clear();
            self.active.clear();
        }
        result
    }
}

impl JSONSafeChunks {
    fn next_chunk(&mut self, py: Python<'_>) -> PyResult<Option<String>> {
        let mut chunk = String::with_capacity(CHUNK_BYTES);
        while chunk.len() < CHUNK_BYTES {
            let Some(frame) = self.frames.pop() else {
                break;
            };
            match frame {
                Frame::Text(text, offset) => {
                    let mut end = (offset + CHUNK_BYTES - chunk.len()).min(text.len());
                    while !text.is_char_boundary(end) {
                        end -= 1;
                    }
                    if end == offset {
                        self.frames.push(Frame::Text(text, offset));
                        break;
                    }
                    chunk.push_str(&text[offset..end]);
                    if end < text.len() {
                        self.frames.push(Frame::Text(text, end));
                    }
                }
                Frame::Value(value, level) => self.push_value(py, value, level)?,
                Frame::Sequence(value, index, len, level) => {
                    // Match Python enumerate when an uncommon child sanitizer
                    // mutates its containing built-in list during traversal.
                    if index < value.bind(py).len()? {
                        let child = value.bind(py).get_item(index)?.unbind();
                        self.frames
                            .push(Frame::Sequence(value, index + 1, len, level));
                        self.frames.push(Frame::Value(child, level));
                        if index > 0 {
                            self.frames
                                .push(Frame::Text(format!(",{}", Self::newline(level)), 0));
                        }
                    }
                }
                Frame::Mapping(value, index, len, level, iterator) => {
                    if index < len {
                        let key = iterator.bind(py).clone().next().transpose()?.unwrap();
                        let child = value.bind(py).get_item(&key)?.unwrap().unbind();
                        let quoted = self.quote.bind(py).call1((&key,))?.extract::<String>()?;
                        self.frames
                            .push(Frame::Mapping(value, index + 1, len, level, iterator));
                        self.frames.push(Frame::Value(child, level));
                        self.frames.push(Frame::Text(
                            format!(
                                "{}{}: ",
                                if index > 0 {
                                    format!(",{}", Self::newline(level))
                                } else {
                                    String::new()
                                },
                                quoted
                            ),
                            0,
                        ));
                    }
                }
                Frame::Close(marker, level, token) => {
                    self.active.remove(&marker);
                    self.frames
                        .push(Frame::Text(format!("{}{token}", Self::newline(level)), 0));
                }
                Frame::Fallback(iterator) => {
                    if let Some(text) = iterator.bind(py).clone().next().transpose()? {
                        self.frames.push(Frame::Fallback(iterator));
                        self.frames.push(Frame::Text(text.extract::<String>()?, 0));
                    }
                }
            }
        }
        Ok(if chunk.is_empty() { None } else { Some(chunk) })
    }
}

#[pyfunction]
pub fn json_safe_chunks(
    py: Python<'_>,
    value: Py<PyAny>,
    fallback: Py<PyAny>,
) -> PyResult<JSONSafeChunks> {
    Ok(JSONSafeChunks {
        frames: vec![Frame::Value(value, 0)],
        active: HashSet::new(),
        fallback,
        quote: py
            .import("json.encoder")?
            .getattr("encode_basestring_ascii")?
            .unbind(),
    })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<JSONSafeChunks>()?;
    module.add_function(wrap_pyfunction!(json_safe_chunks, module)?)?;
    Ok(())
}
