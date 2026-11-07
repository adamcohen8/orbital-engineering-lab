use std::{env,path::PathBuf,process::Command};
fn run(command: &mut Command) {
    assert!(command.status().expect("CMake/C compiler unavailable").success(), "SDK C build failed");
}
fn main() {
    let source = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../c");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("c-sdk");
    run(Command::new("cmake").arg("-S").arg(&source).arg("-B").arg(&out));
    run(Command::new("cmake").arg("--build").arg(&out).args(["--config","Release","--parallel","1"]));
    println!("cargo:rustc-link-search=native={}",out.display());
    println!("cargo:rustc-link-search=native={}",out.join("Release").display());
    println!("cargo:rustc-link-lib=static=oel_fsw_sdk");
    if env::var("CARGO_CFG_TARGET_FAMILY").unwrap()=="unix" { println!("cargo:rustc-link-lib=m"); }
    println!("cargo:rerun-if-changed={}",source.display());
}
