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
pub(crate) use format::fmt_f12;
pub(crate) use green::{write_less_section, write_lmix_section, write_mat_section, write_ret_section};

pub use green::{
    contour_from_str, contour_to_string, green_from_str, green_partial_from_str,
    green_to_string, less_from_str, less_to_string, lmix_from_str, lmix_to_string,
    mat_from_str, mat_to_string, read_contour, ret_from_str, ret_to_string,
    time_function_from_str, time_function_to_string, timestep_from_str, timestep_to_string,
};
pub use moving::{
    cf_moving_read_into, cf_moving_to_string, moving_green_read_into, moving_green_to_string,
    read_cf_moving, read_moving_green, write_cf_moving, write_moving_green,
};
pub use ness::{ness_array_from_str, ness_array_to_string, ness_from_str, ness_to_string};
pub use pseudo::{
    pseudo_green_read_into, pseudo_green_to_string, read_pseudo_green_into, write_pseudo_green,
};
