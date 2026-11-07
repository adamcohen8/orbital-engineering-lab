//! Public OEL SDK. Borrows request data; C handles bounded framing and JSON.
//! All FSW behavior belongs in the implementing Rust FlightSoftware type.
use std::{ffi::{c_char,c_int,c_void,CStr,CString},panic::{catch_unwind,AssertUnwindSafe}};
#[repr(C)] struct Json { _private: [u8;0] }
extern "C" {
    fn oel_run(callback: extern "C" fn(*const c_char,*const Json,*mut c_char,usize,*mut c_void)->c_int,
               context:*mut c_void,checkpointable:c_int)->c_int;
    fn oel_field(json:*const Json,object:c_int,key:*const c_char)->c_int;
    fn oel_find_schema(json:*const Json,schema:*const c_char)->c_int;
    fn oel_vector(json:*const Json,object:c_int,key:*const c_char,out:*mut f64,count:usize)->c_int;
    fn oel_raw(json:*const Json,token:c_int,out:*mut c_char,capacity:usize)->c_int;
    fn oel_wrench(json:*const Json,boot:*const c_char,force:*const f64,torque:*const f64,
                  out:*mut c_char,capacity:usize)->c_int;
}
/// The view cannot outlive the callback; raw host pointers are never exposed.
pub struct Input<'a> { raw:*const Json, _borrow:std::marker::PhantomData<&'a ()> }
impl<'a> Input<'a> {
    pub fn find_schema(&self,name:&str)->Option<i32> {
        let name=CString::new(name).ok()?;
        let token=unsafe{oel_find_schema(self.raw,name.as_ptr())};
        if token<0 {None} else {Some(token)}
    }
    pub fn vector<const N:usize>(&self,object:i32,key:&str)->Option<[f64;N]> {
        if object<0{return None;} let key=CString::new(key).ok()?;let mut out=[0.;N];
        if unsafe{oel_vector(self.raw,object,key.as_ptr(),out.as_mut_ptr(),N)}==0 {Some(out)} else {None}
    }
    pub fn field(&self,key:&str)->Result<String,String> {
        let key=CString::new(key).map_err(|e|e.to_string())?;
        let token=unsafe{oel_field(self.raw,-1,key.as_ptr())};
        if token<0{return Err("Missing field".into());}
        let mut out=vec![0u8;262144];
        if unsafe{oel_raw(self.raw,token,out.as_mut_ptr().cast(),out.len())}!=0{return Err("Field too large".into());}
        Ok(unsafe{CStr::from_ptr(out.as_ptr().cast())}.to_string_lossy().into_owned())
    }
    pub fn raw_json(&self)->Result<String,String> {
        let mut out=vec![0u8;262144];
        if unsafe{oel_raw(self.raw,-1,out.as_mut_ptr().cast(),out.len())}!=0{return Err("Input too large".into());}
        Ok(unsafe{CStr::from_ptr(out.as_ptr().cast())}.to_string_lossy().into_owned())
    }
    pub fn wrench(&self,boot_id_json:&str,force:[f64;3],torque:[f64;3])->Result<String,String> {
        let boot=CString::new(boot_id_json).map_err(|e|e.to_string())?;
        let mut out=vec![0u8;262144];
        if unsafe{oel_wrench(self.raw,boot.as_ptr(),force.as_ptr(),torque.as_ptr(),out.as_mut_ptr().cast(),out.len())}!=0 {
            return Err("Invalid wrench or identity".into());
        }
        Ok(unsafe{CStr::from_ptr(out.as_ptr().cast())}.to_string_lossy().into_owned())
    }
}
pub trait FlightSoftware {
    /// Return a JSON boundary payload. Err terminates the host transaction.
    fn on_message(&mut self,operation:&str,input:Input<'_>)->Result<String,String>;
    fn checkpointable(&self)->bool {false}
}
pub fn run<T:FlightSoftware>(software:&mut T)->i32 {
    extern "C" fn dispatch<T:FlightSoftware>(op:*const c_char,input:*const Json,out:*mut c_char,
                                            capacity:usize,context:*mut c_void)->c_int {
        let result=catch_unwind(AssertUnwindSafe(||{
            let software=unsafe{&mut *context.cast::<T>()};
            let operation=unsafe{CStr::from_ptr(op)}.to_str().map_err(|e|e.to_string())?;
            software.on_message(operation,Input{raw:input,_borrow:std::marker::PhantomData})
        }));
        match result {
            Ok(Ok(text)) if text.len()<capacity && !text.contains('\0')=>{
                unsafe{std::ptr::copy_nonoverlapping(text.as_ptr(),out.cast(),text.len());*out.add(text.len())=0;} 0
            }, _=>-1
        }
    }
    unsafe{oel_run(dispatch::<T>,(software as *mut T).cast(),software.checkpointable() as c_int)}
}
