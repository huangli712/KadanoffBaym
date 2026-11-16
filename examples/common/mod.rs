//
// Shared helpers of the tutorial examples. Included by every example and
// validation binary through `#[path = "common/mod.rs"] mod common;`.
//

// shared by several example binaries; not every binary uses every item
// Shared example sources are included by several binaries.
#![allow(clippy::duplicate_mod)]

#[allow(dead_code)]
pub mod config;
#[allow(dead_code)]
pub mod contour_loop;
#[allow(dead_code)]
pub mod kbref;
#[path = "../../util/kbref.rs"]
pub mod kbref_raw;
pub mod ness_util;
#[allow(dead_code)]
pub mod output;
#[allow(dead_code)]
pub mod toml;
