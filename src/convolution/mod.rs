// Project : KadanoffBaym
// File    : src/convolution/mod.rs
// Purpose : convolution on the contour: the kernels with their full and
//           incremental drivers, the Matsubara and pseudo-particle
//           variants, the density matrices, the response function and
//           the bubbles

mod bubble;
mod check;
mod density;
mod full;
mod incr;
mod less;
mod lmix;
mod mat;
mod pseudo;
mod response;
mod ret;
mod workspace;

pub use workspace::{ConvWorkspace, FOp};

pub use bubble::{bubble1, bubble1_herm};
pub use bubble::{bubble1_moving, bubble1_moving_herm, bubble1_steady, bubble1_steady_scalar};
pub use bubble::{bubble2, bubble2_herm};
pub use bubble::{bubble2_moving, bubble2_moving_herm, bubble2_steady, bubble2_steady_scalar};
pub use density::{convolution_density_matrix, steady_convolution_density_matrix};
pub use density::{convolution_les_timediag, convolution_les_timediag_herm};
pub use full::{convolution, convolution_ab, convolution_timestep};
pub use incr::{incr_convolution, incr_convolution_step};
pub use less::{incr_convolution_less, incr_convolution_less_adv, incr_convolution_less_step};
pub use less::{incr_convolution_lmix_rmix, incr_convolution_ret_less};
pub use less::{incr_less_adv_buf, incr_lmix_rmix_buf};
pub use lmix::{conv_lmix_mat, conv_ret_lmix_kernel, incr_convolution_lmix};
pub use mat::{conv_mat, incr_convolution_mat};
pub use pseudo::{pseudo_convolution, pseudo_convolution_mat, pseudo_convolution_timestep};
pub use pseudo::{pseudo_incr_convolution, pseudo_incr_convolution_step};
pub use pseudo::{pseudo_incr_convolution_lmix, pseudo_incr_convolution_mat};
pub use response::response_convolution;
pub use ret::incr_convolution_ret;

pub(crate) use density::convolution_density_matrix_herm;
pub(crate) use less::incr_less_diagcc;
pub(crate) use less::{conv_less_adv_kernel, conv_lmix_rmix_kernel, conv_ret_less_kernel};
pub(crate) use less::{incr_convolution_less_adv_step, incr_convolution_lmix_rmix_step};
pub(crate) use lmix::{conv_lmix_mat_loop, incr_convolution_lmix_step};
pub(crate) use mat::{conv_mat_at, f0_matrix};
pub(crate) use pseudo::pseudo_incr_convolution_lmix_step;
pub(crate) use ret::{conv_ret_row, incr_convolution_ret_step};
pub(crate) use workspace::{a_times_f, f_times_b};
