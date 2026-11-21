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

pub use green::green_partial_from_str;
pub use green::read_contour;
pub use moving::{moving_green_from_str, moving_time_function_from_str};
pub use moving::{read_moving_green, write_moving_green};
pub use moving::{read_moving_time_function, write_moving_time_function};
pub use pseudo::{pseudo_green_from_str, read_pseudo_green, write_pseudo_green};

pub(crate) use lines::Lines;

pub(crate) use format::{fmt_e16, fmt_f12, fmt_g17, fmt_i4};
pub(crate) use green::write_less_section;
pub(crate) use green::write_lmix_section;
pub(crate) use green::write_mat_section;
pub(crate) use green::write_ret_section;
