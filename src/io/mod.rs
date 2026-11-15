//
// Project : KadanoffBaym
// File    : src/io/mod.rs
// Purpose : plain-text serialization of the core containers (standard,
//           moving, NESS, pseudo-particle), split by container flavor
//

mod error;
mod format;
mod green;
mod moving;
mod ness;
mod pseudo;

pub use error::IoError;

pub use green::read_contour;
pub use green::{contour_from_str, contour_to_string};
pub use green::{green_from_str, green_partial_from_str, green_to_string};
pub use green::{less_from_str, less_to_string};
pub use green::{lmix_from_str, lmix_to_string};
pub use green::{mat_from_str, mat_to_string};
pub use green::{ret_from_str, ret_to_string};
pub use green::{time_function_from_str, time_function_to_string};
pub use green::{timestep_from_str, timestep_to_string};

pub use moving::{
    cf_moving_read_into, cf_moving_to_string, moving_green_read_into, moving_green_to_string,
    read_cf_moving, read_moving_green, write_cf_moving, write_moving_green,
};

pub use ness::{ness_array_from_str, ness_array_to_string};
pub use ness::{ness_from_str, ness_to_string};

pub use pseudo::{
    pseudo_green_from_str, pseudo_green_to_string, read_pseudo_green, write_pseudo_green,
};

pub(crate) use format::{fmt_f12, fmt_i4};
pub(crate) use green::write_less_section;
pub(crate) use green::write_lmix_section;
pub(crate) use green::write_mat_section;
pub(crate) use green::write_ret_section;
