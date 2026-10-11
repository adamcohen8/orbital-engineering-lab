//! Trusted internal evidence conversion. Never replaces boundary validation.
use pyo3::exceptions::{PyRecursionError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType};
use std::{collections::HashMap, sync::Arc};

struct Metadata {
    _owner: Py<PyType>,
    qualified: String,
    names: Option<Vec<Py<PyString>>>,
}

#[pyclass]
pub struct TrustedPacketEncoder {
    classifier: Py<PyAny>,
    enum_type: Py<PyAny>,
    mapping_type: Py<PyAny>,
    base64: Py<PyAny>,
    metadata: HashMap<usize, Arc<Metadata>>,
}

impl TrustedPacketEncoder {
    fn walk<'py>(
        &mut self,
        value: &Bound<'py, PyAny>,
        depth: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        if depth > 900 {
            return Err(PyRecursionError::new_err(
                "maximum recursion depth exceeded",
            ));
        }
        if value.is_none()
            || value.is_instance_of::<PyString>()
            || value.is_instance_of::<PyBool>()
            || value.is_instance_of::<PyInt>()
        {
            return Ok(value.clone());
        }
        if value.is_instance_of::<PyFloat>() {
            if !value.extract::<f64>()?.is_finite() {
                return Err(PyValueError::new_err(
                    "boundary floating-point values must be finite",
                ));
            }
            return Ok(value.clone());
        }
        let py = value.py();
        if value.is_instance_of::<PyBytes>() {
            let result = PyDict::new(py);
            result.set_item(
                "$bytes_base64",
                self.base64
                    .bind(py)
                    .call1((value,))?
                    .call_method1("decode", ("ascii",))?,
            )?;
            return Ok(result.into_any());
        }
        // Exact built-in containers cannot be enums, dataclasses or custom
        // mappings. Keep those reflective checks for all external subclasses.
        if value.is_exact_instance_of::<PyDict>() {
            let result = PyDict::new(py);
            for (key, item) in value.cast::<PyDict>()?.iter() {
                if !key.is_instance_of::<PyString>() {
                    return Err(PyTypeError::new_err(
                        "boundary mapping keys must be strings",
                    ));
                }
                result.set_item(key, self.walk(&item, depth + 1)?)?;
            }
            return Ok(result.into_any());
        }
        if value.is_exact_instance_of::<PyList>() || value.is_exact_instance_of::<PyTuple>() {
            let result = PyList::empty(py);
            for item in value.try_iter()? {
                result.append(self.walk(&item?, depth + 1)?)?;
            }
            return Ok(result.into_any());
        }
        if value.is_instance(self.enum_type.bind(py))? {
            return self.walk(&value.getattr("value")?, depth + 1);
        }
        let kind = value.get_type();
        let key = kind.as_ptr() as usize;
        let info = match self.metadata.get(&key) {
            Some(info) => Arc::clone(info),
            None => {
                let result = self.classifier.bind(py).call1((&kind,))?;
                let (qualified, names): (String, Option<Vec<Py<PyString>>>) = result.extract()?;
                let info = Arc::new(Metadata {
                    _owner: kind.unbind(),
                    qualified,
                    names,
                });
                self.metadata.insert(key, Arc::clone(&info));
                info
            }
        };
        if let Some(names) = &info.names {
            if !value.is_instance_of::<PyType>() {
                let result = PyDict::new(py);
                for name in names {
                    let name = name.bind(py);
                    result.set_item(name, self.walk(&value.getattr(name)?, depth + 1)?)?;
                }
                return Ok(result.into_any());
            }
        }
        if value.is_instance(self.mapping_type.bind(py))? {
            let result = PyDict::new(py);
            for item in value.call_method0("items")?.try_iter()? {
                let item = item?;
                let key = item.get_item(0)?;
                if !key.is_instance_of::<PyString>() {
                    return Err(PyTypeError::new_err(
                        "boundary mapping keys must be strings",
                    ));
                }
                result.set_item(key, self.walk(&item.get_item(1)?, depth + 1)?)?;
            }
            return Ok(result.into_any());
        }
        if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
            let result = PyList::empty(py);
            for item in value.try_iter()? {
                result.append(self.walk(&item?, depth + 1)?)?;
            }
            return Ok(result.into_any());
        }
        Err(PyTypeError::new_err(format!(
            "unsupported boundary value type {}",
            info.qualified
        )))
    }
}

#[pymethods]
impl TrustedPacketEncoder {
    #[new]
    fn new(py: Python<'_>, classifier: Py<PyAny>) -> PyResult<Self> {
        Ok(Self {
            classifier,
            enum_type: py.import("enum")?.getattr("Enum")?.unbind(),
            mapping_type: py.import("collections.abc")?.getattr("Mapping")?.unbind(),
            base64: py.import("base64")?.getattr("b64encode")?.unbind(),
            metadata: HashMap::new(),
        })
    }
    fn convert<'py>(&mut self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.walk(value, 0)
    }
    fn convert_many<'py>(&mut self, values: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
        let result = PyList::empty(values.py());
        for value in values.try_iter()? {
            result.append(self.walk(&value?, 0)?)?;
        }
        Ok(result)
    }
}
