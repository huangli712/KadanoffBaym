//
// Shared helpers of the tutor examples (the Rust counterpart of
// julia/tutor/common/). Included by every example and validation
// binary through `#[path = "common/mod.rs"] mod common;`.
//

// shared by several example binaries; not every binary uses every item
#[allow(dead_code)]
pub mod config;
#[allow(dead_code)]
pub mod contour_loop;
#[path = "../../tools/kbref.rs"]
#[allow(dead_code)]
pub mod kbref_raw;
#[allow(dead_code)]
pub mod kbref;
#[allow(dead_code)]
pub mod output;
#[allow(dead_code)]
pub mod profile;
#[allow(dead_code)]
pub mod toml;
