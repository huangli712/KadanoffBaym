//
// Project : KadanoffBaym
// File    : src/io/mod.rs
// Purpose : plain-text serialization of the core containers (standard,
//           moving, NESS, pseudo-particle), split by container flavor
//

mod error;
mod format;
mod green;
mod lines;
mod moving;
mod pseudo;
mod steady;

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
pub use moving::{moving_green_from_str, moving_green_to_string};
pub use moving::{moving_time_function_from_str, moving_time_function_to_string};
pub use moving::{read_moving_green, write_moving_green};
pub use moving::{read_moving_time_function, write_moving_time_function};
pub use pseudo::{pseudo_green_from_str, pseudo_green_to_string};
pub use pseudo::{read_pseudo_green, write_pseudo_green};
pub use steady::{steady_function_from_str, steady_function_to_string};
pub use steady::{steady_green_from_str, steady_green_to_string};

pub(crate) use lines::Lines;

pub(crate) use format::{fmt_e16, fmt_f12, fmt_i4};
pub(crate) use green::write_less_section;
pub(crate) use green::write_lmix_section;
pub(crate) use green::write_mat_section;
pub(crate) use green::write_ret_section;
