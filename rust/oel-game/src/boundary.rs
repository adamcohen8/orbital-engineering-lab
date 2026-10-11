//! Native traversal of maintained boundary records; Python owns external wrappers.
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
pyo3::create_exception!(oel_rust_game, NeedsPythonValidation, PyException);
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PySet, PyString, PyTuple, PyType,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

struct Record {
    _owner: Py<PyType>,
    fields: Vec<(Py<PyString>, Py<PyString>)>,
}

#[pyclass]
pub struct BoundaryValidator {
    records: HashMap<usize, Arc<Record>>,
    leaves: Py<PyAny>,
    forbidden_keys: Py<PySet>,
    python_walk: Py<PyAny>,
    key_error: Py<PyAny>,
    format_key: Py<PyAny>,
}

impl BoundaryValidator {
    fn leaf(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(value.is_none() || value.is_instance(self.leaves.bind(value.py()))?)
    }
    fn fallback(
        &self,
        value: &Bound<'_, PyAny>,
        path: &[Py<PyString>],
        seen: &mut HashSet<usize>,
    ) -> PyResult<()> {
        let py = value.py();
        let paths = PyList::new(py, path.iter().map(|s| s.bind(py)))?;
        let identities = PySet::new(py, seen.iter().copied())?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("path", paths)?;
        kwargs.set_item("seen", &identities)?;
        self.python_walk.bind(py).call((value,), Some(&kwargs))?;
        for item in identities.iter() {
            seen.insert(item.extract()?);
        }
        Ok(())
    }
    fn walk(
        &self,
        value: &Bound<'_, PyAny>,
        path: &mut Vec<Py<PyString>>,
        seen: &mut HashSet<usize>,
        depth: usize,
    ) -> PyResult<()> {
        if value.is_none() {
            return Ok(());
        }
        let record = self.records.get(&(value.get_type().as_ptr() as usize));
        if record.is_none()
            && !value.is_exact_instance_of::<PyDict>()
            && !value.is_exact_instance_of::<PyList>()
            && !value.is_exact_instance_of::<PyTuple>()
            && !value.is_exact_instance_of::<PyString>()
            && !value.is_exact_instance_of::<PyBool>()
            && !value.is_exact_instance_of::<PyInt>()
            && !value.is_exact_instance_of::<PyFloat>()
            && !value.is_exact_instance_of::<PyBytes>()
        {
            // Delegate before predicate checks, which may access __class__ on
            // external objects. Python owns their traversal and diagnostics.
            return self.fallback(value, path, seen);
        }
        if self.leaf(value)? {
            return Ok(());
        }
        let identity = value.as_ptr() as usize;
        if seen.contains(&identity) {
            return Ok(());
        }
        let py = value.py();
        if depth > 128 {
            return Err(NeedsPythonValidation::new_err(()));
        }
        seen.insert(identity);
        if let Some(record) = record {
            for (name, component) in &record.fields {
                let child = value.getattr(name.bind(py))?;
                if self.leaf(&child)? {
                    continue;
                }
                path.push(component.clone_ref(py));
                self.walk(&child, path, seen, depth + 1)?;
                path.pop();
            }
            return Ok(());
        }
        if value.is_exact_instance_of::<PyDict>() {
            // Python's items iterator retains mutation detection and ordering.
            for pair in value.call_method0("items")?.try_iter()? {
                let pair = pair?;
                let key = pair.get_item(0)?;
                let child = pair.get_item(1)?;
                let lower = key.str()?.call_method0("lower")?;
                if self.forbidden_keys.bind(py).contains(lower)? {
                    let paths = PyList::new(py, path.iter().map(|s| s.bind(py)))?;
                    self.key_error.bind(py).call1((paths, &key))?;
                }
                if self.leaf(&child)? {
                    continue;
                }
                // Python formatting preserves Unicode and custom key formatting.
                let component = self.format_key.bind(py).call1((&key, ""))?;
                let component = PyString::new(py, ".")
                    .call_method1("__add__", (component,))?
                    .cast_into::<PyString>()?;
                path.push(component.unbind());
                self.walk(&child, path, seen, depth + 1)?;
                path.pop();
            }
            return Ok(());
        }
        for (index, child) in value.try_iter()?.enumerate() {
            let child = child?;
            if self.leaf(&child)? {
                continue;
            }
            path.push(PyString::new(py, &format!("[{index}]")).unbind());
            self.walk(&child, path, seen, depth + 1)?;
            path.pop();
        }
        Ok(())
    }
}

#[pymethods]
impl BoundaryValidator {
    #[new]
    fn new(
        py: Python<'_>,
        records: &Bound<'_, PyDict>,
        leaves: Py<PyAny>,
        forbidden_keys: Py<PySet>,
        python_walk: Py<PyAny>,
        key_error: Py<PyAny>,
    ) -> PyResult<Self> {
        let mut cache = HashMap::new();
        for (kind, names) in records.iter() {
            let kind = kind.cast_into::<PyType>()?;
            let mut fields = Vec::new();
            for name in names.try_iter()? {
                let name = name?.cast_into::<PyString>()?;
                let component = PyString::new(py, ".")
                    .call_method1("__add__", (&name,))?
                    .cast_into::<PyString>()?;
                fields.push((name.unbind(), component.unbind()));
            }
            cache.insert(
                kind.as_ptr() as usize,
                Arc::new(Record {
                    _owner: kind.unbind(),
                    fields,
                }),
            );
        }
        Ok(Self {
            records: cache,
            leaves,
            forbidden_keys,
            python_walk,
            key_error,
            format_key: py.import("builtins")?.getattr("format")?.unbind(),
        })
    }
    fn check_many(&self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        for value in values.try_iter()? {
            self.check(&value?)?;
        }
        Ok(())
    }
    fn check(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = vec![PyString::new(value.py(), "$").unbind()];
        match self.walk(
            value,
            &mut path.iter().map(|s| s.clone_ref(value.py())).collect(),
            &mut HashSet::new(),
            0,
        ) {
            Err(error) if error.is_instance_of::<NeedsPythonValidation>(value.py()) => {
                self.fallback(value, &path, &mut HashSet::new())
            }
            result => result,
        }
    }
}
