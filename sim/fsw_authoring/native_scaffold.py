"""Editable C, C++, and Rust starters sharing the public SDK and wire protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .scaffold import CandidateScaffoldResult, _scenario, _slug, _write_new

C_SOURCE = r"""#include "oel_fsw.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
@CPP_INCLUDE@
typedef struct { char boot[1024]; double q[4],w[3],position[3],velocity[3],target[3],target_v[3]; int own,track; } State;
static void limit(double v[3],double maximum) {
    double norm=sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);int k;
    if(norm>maximum) for(k=0;k<3;++k) v[k]*=maximum/norm;
}
static int step(const char *operation,const oel_json *input,char *out,size_t capacity,void *context) {
    State *s=(State*)context;int own,track,k; double f[3]={0,0,0},t[3]={0,0,0};
    if(!strcmp(operation,"boot")) {
        int boot=oel_field(input,-1,"boot_id");
        if(boot<0 || oel_raw(input,boot,s->boot,sizeof s->boot)) return -1;
        return snprintf(out,capacity,"{}")>=2?0:-1;
    }
    if(!strcmp(operation,"shutdown")) {snprintf(out,capacity,"{}");return 0;}
    if(strcmp(operation,"step")) return -1;
    own=oel_find_schema(input,"ideal_own_state.v1");
    track=oel_find_schema(input,"ideal_tracked_object_state.v1");
    if(own>=0) {
        oel_vector(input,own,"attitude_quat_body_from_inertial",s->q,4);
        oel_vector(input,own,"angular_rate_body_rad_s",s->w,3);
        if(!oel_vector(input,own,"position_m",s->position,3) && !oel_vector(input,own,"velocity_m_s",s->velocity,3)) s->own=1;
    }
    if(track>=0 && !oel_vector(input,track,"position_m",s->target,3) && !oel_vector(input,track,"velocity_m_s",s->target_v,3)) s->track=1;
    if(@RPO@) {
        if(s->own&&s->track) {
            double a[3],cross[3],cross2[3];
            for(k=0;k<3;++k) a[k]=-0.02*(s->position[k]-s->target[k])-4.0*(s->velocity[k]-s->target_v[k]);
            for(k=0;k<3;++k) cross[k]=s->q[1+(k+1)%3]*a[(k+2)%3]-s->q[1+(k+2)%3]*a[(k+1)%3];
            for(k=0;k<3;++k) cross2[k]=s->q[1+(k+1)%3]*cross[(k+2)%3]-s->q[1+(k+2)%3]*cross[(k+1)%3];
            for(k=0;k<3;++k) f[k]=a[k]+2*s->q[0]*cross[k]+2*cross2[k];
            limit(f,10.0);
        }
    } else {
        for(k=0;k<3;++k) t[k]=-0.02*(s->q[0]<0?-1:1)*s->q[k+1]-0.1*s->w[k];
        limit(t,0.05);
    }
    return oel_wrench(input,s->boot,f,t,out,capacity);
}
@MAIN@
"""
CPP_MAIN = r"""
class Candidate final : public oel::FlightSoftware {
    State state{};
public:
    Candidate() { state.q[0]=1; }
    int on_message(const std::string& operation,oel::Input input,char *out,size_t capacity) override {
        return step(operation.c_str(),input.raw,out,capacity,&state);
    }
};
int main(int argc,char **argv) {
    if(argc==2 && !strcmp(argv[1],"--self-test")) {double v[3]={3,4,0};limit(v,1);return fabs(v[0]-0.6)<1e-12?0:1;}
    Candidate software; return software.run();
}
"""
C_MAIN = r"""
int main(int argc,char **argv) {
    State state;memset(&state,0,sizeof state);state.q[0]=1;
    if(argc==2 && !strcmp(argv[1],"--self-test")) {double v[3]={3,4,0};limit(v,1);return fabs(v[0]-0.6)<1e-12?0:1;}
    return oel_run(step,&state,0);
}
"""
RUST_SOURCE = r"""use oel_fsw_sdk::{FlightSoftware,Input};
struct Candidate {boot:String,q:[f64;4],w:[f64;3],position:[f64;3],velocity:[f64;3],target:[f64;3],target_v:[f64;3],own:bool,track:bool}
fn limit(v:&mut [f64;3],maximum:f64) {let n=v.iter().map(|x|x*x).sum::<f64>().sqrt();if n>maximum{for x in v{*x*=maximum/n;}}}
impl FlightSoftware for Candidate {
    fn on_message(&mut self,operation:&str,input:Input<'_>)->Result<String,String> {
        match operation {
            "boot"=>{self.boot=input.field("boot_id")?;return Ok("{}".into());},
            "shutdown"=>return Ok("{}".into()),
            "step"=>{},_=>return Err("Unsupported lifecycle operation".into())
        }
        if let Some(own)=input.find_schema("ideal_own_state.v1") {
            if let Some(q)=input.vector(own,"attitude_quat_body_from_inertial"){self.q=q;}
            if let Some(w)=input.vector(own,"angular_rate_body_rad_s"){self.w=w;}
            if let (Some(p),Some(v))=(input.vector(own,"position_m"),input.vector(own,"velocity_m_s")){self.position=p;self.velocity=v;self.own=true;}
        }
        if let Some(track)=input.find_schema("ideal_tracked_object_state.v1") {
            if let (Some(p),Some(v))=(input.vector(track,"position_m"),input.vector(track,"velocity_m_s")){self.target=p;self.target_v=v;self.track=true;}
        }
        let(mut f,mut t)=([0.;3],[0.;3]);
        if @RPO@ {
            if self.own&&self.track {
                let mut a=[0.;3];let mut cross=[0.;3];let mut cross2=[0.;3];
                for k in 0..3{a[k]=-0.02*(self.position[k]-self.target[k])-4.0*(self.velocity[k]-self.target_v[k]);}
                for k in 0..3{cross[k]=self.q[1+(k+1)%3]*a[(k+2)%3]-self.q[1+(k+2)%3]*a[(k+1)%3];}
                for k in 0..3{cross2[k]=self.q[1+(k+1)%3]*cross[(k+2)%3]-self.q[1+(k+2)%3]*cross[(k+1)%3];}
                for k in 0..3{f[k]=a[k]+2.*self.q[0]*cross[k]+2.*cross2[k];}limit(&mut f,10.);
            }
        } else {
            for k in 0..3{t[k]=-0.02*(if self.q[0]<0.{-1.}else{1.})*self.q[k+1]-0.1*self.w[k];}limit(&mut t,0.05);
        }
        input.wrench(&self.boot,f,t)
    }
}
fn main(){let mut s=Candidate{boot:String::new(),q:[1.,0.,0.,0.],w:[0.;3],position:[0.;3],velocity:[0.;3],target:[0.;3],target_v:[0.;3],own:false,track:false};std::process::exit(oel_fsw_sdk::run(&mut s));}
#[test]fn limits_vector_norm(){let mut v=[3.,4.,0.];limit(&mut v,1.);assert!((v[0]-0.6).abs()<1e-12);}
"""


def scaffold_native(
    name: str,
    *,
    language: str,
    template: str,
    workspace_root: Path,
    output_dir: str | Path | None = None,
    force: bool = False,
    writer: Any | None = None,
) -> CandidateScaffoldResult:
    from .candidate import inspect_candidate

    root = workspace_root.resolve()
    candidate_id = _slug(name)
    destination = Path(output_dir) if output_dir else root / "fsw_candidates" / candidate_id
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    destination.relative_to(root)
    if writer is not None and Path(
        getattr(writer, "logical_root", writer.path)
    ).expanduser().resolve() != destination:
        raise PermissionError("Native scaffold writer must be bound to the logical candidate output directory.")
    if template not in {"adcs", "rpo"} or language not in {"c", "cpp", "rust"}:
        raise ValueError("Native language must be c, cpp, or rust; template must be adcs or rpo.")
    # Check the full proposed tree before writing even one file.
    manifest = destination / "candidate.yaml"
    scenario_path = destination / "configs" / "smoke.yaml"
    scenario = _scenario(
        candidate_id=candidate_id,
        module_path="sim.fsw_authoring.native_transport",
        class_name="NativeFlightSoftwareAdapter",
        template=template,
    )
    controlled = scenario["objects"]["target" if template == "adcs" else "chaser"]
    controlled["flight_software"]["params"].update(manifest_path=str(manifest), workspace_root=str(root))
    scenario["outputs"]["stats"]["save_full_log"] = False
    config = {
        "schema_version": "oel.fsw_authoring.candidate.v2",
        "candidate_id": candidate_id,
        "revision": "0.1.0",
        "kind": "native_stack",
        "source": {"root": "src", "revision_id": "editable-public-starter", "language": language},
        "interfaces": {
            "onboard_contract": "oel.fsw.boundary.v1",
            "hardware_profile": "hardware.ideal_wrench.v1",
            "task_period_s": 0.2 if template == "adcs" else 0.5,
        },
        "claims": {"intended_use": f"educational_{template}"},
        "verification": {"component_suite": "src", "smoke_case": "configs/smoke.yaml"},
        "handling": {"classification": "public", "hosted_ai_allowed": False},
        "native": {"checkpointable": False, "timeout_s": 5.0},
    }
    files = {
        manifest: yaml.safe_dump(config, sort_keys=False),
        scenario_path: yaml.safe_dump(scenario, sort_keys=False),
    }
    sdk = Path(__file__).with_name("sdk")
    files[destination / "src" / "sdk" / "LICENSE.txt"] = (sdk / "LICENSE.txt").read_text()
    sdk_files = ["c/oel_fsw.h", "c/oel_fsw.c", "c/CMakeLists.txt"]
    if language == "cpp":
        sdk_files.append("cpp/oel_fsw.hpp")
    if language == "rust":
        sdk_files.extend(["rust/Cargo.toml", "rust/build.rs", "rust/src/lib.rs"])
    for relative in sdk_files:
        path = sdk / relative
        if path.is_symlink():
            raise PermissionError("SDK source files may not be symlinks.")
        files[destination / "src" / "sdk" / relative] = path.read_text()
    if language == "rust":
        files[destination / "src" / "Cargo.toml"] = """[package]
name = "oel_candidate"
version = "0.1.0"
edition = "2021"
[dependencies]
oel-fsw-sdk = { path = "sdk/rust" }
[[bin]]
name = "oel_candidate"
path = "main.rs"
"""
        files[
            destination / "src" / "Cargo.lock"
        ] = """# Generated dependency lock for the dependency-free public starter.
version = 4
[[package]]
name = "oel-fsw-sdk"
version = "0.1.0"
[[package]]
name = "oel_candidate"
version = "0.1.0"
dependencies = ["oel-fsw-sdk"]
"""
        files[destination / "src" / "main.rs"] = RUST_SOURCE.replace("@RPO@", "true" if template == "rpo" else "false")
    else:
        extension = "cpp" if language == "cpp" else "c"
        files[destination / "src" / f"main.{extension}"] = (
            C_SOURCE.replace("@RPO@", "1" if template == "rpo" else "0")
            .replace("@CPP_INCLUDE@", '#include "oel_fsw.hpp"' if language == "cpp" else "")
            .replace("@MAIN@", CPP_MAIN if language == "cpp" else C_MAIN)
        )
        files[destination / "src" / "CMakeLists.txt"] = f"""cmake_minimum_required(VERSION 3.20)
project(oel_candidate LANGUAGES C {"CXX" if language == "cpp" else ""})
add_subdirectory(sdk/c)
add_executable(oel_candidate main.{extension})
target_link_libraries(oel_candidate PRIVATE oel_fsw_sdk)
target_include_directories(oel_candidate PRIVATE sdk/cpp)
set_target_properties(oel_candidate PROPERTIES CXX_STANDARD 17 RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/bin")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(oel_candidate PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${{config}} "${{CMAKE_BINARY_DIR}}/bin")
endforeach()
enable_testing()
add_test(NAME component COMMAND oel_candidate --self-test)
"""
    files[destination / "README.md"] = f"""# {name}: {language} {template} starter

Edit src/main.{"rs" if language == "rust" else language}. OEL owns the physics.
This ideal-measurement starter is educational simulation FSW, not flight qualification.
The RPO starter is bounded ECI position/velocity feedback, not a mission planner.

From the authorized workspace:

    oel fsw inspect {manifest.relative_to(root)}
    oel fsw plan {manifest.relative_to(root)} build
    oel fsw build {manifest.relative_to(root)} --trusted-build
    oel fsw validate {manifest.relative_to(root)}
    oel fsw test {manifest.relative_to(root)}
    oel fsw smoke {manifest.relative_to(root)}

Build scripts and executables must be trusted. Stdio is not a security sandbox.
CMake and a C compiler are required; C++ also needs C++, Rust also needs Cargo/rustc.
Starters need no downloads or registry dependencies. Checkpoints are explicitly disabled.
"""
    for path, content in files.items():
        path.resolve().relative_to(root)
        if path.is_symlink() or path.exists() and path.read_text() != content and not force:
            raise FileExistsError(f"Refusing to overwrite native scaffold: {path}")
    written = [
        path
        for path, content in files.items()
        if _write_new(path, content, force=force, writer=writer, logical_root=destination)
    ]
    if writer is None or Path(writer.path).expanduser().resolve() == destination:
        inspection_manifest = manifest
        inspection_root = root
    else:
        writer_root = Path(writer.path).expanduser().resolve()
        logical_root = Path(getattr(writer, "logical_root", destination)).expanduser().resolve()
        inspection_manifest = writer_root / manifest.relative_to(logical_root)
        inspection_root = writer_root
    return CandidateScaffoldResult(
        destination, manifest, tuple(sorted(written)), inspect_candidate(inspection_manifest, workspace_root=inspection_root)
    )
