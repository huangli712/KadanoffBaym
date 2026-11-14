// Project : KadanoffBaym
// File    : src/convolution/mod.rs
// Purpose : real-time convolution kernels (blueprint: julia/src/convolution/)

mod bubble;
mod density_matrix;
mod incr;
mod less;
mod lmix;
mod mat;
mod pseudo;
mod response;
mod ret;
mod timestep;
mod workspace;

pub use bubble::{
    bubble1, bubble1_herm, bubble1_moving, bubble1_moving_herm, bubble1_ness, bubble1_ness_scalar,
    bubble2, bubble2_herm, bubble2_moving, bubble2_moving_herm, bubble2_ness, bubble2_ness_scalar,
};
pub use density_matrix::{
    convolution_density_matrix, convolution_les_timediag, convolution_les_timediag_herm,
};
pub use incr::{incr_convolution, incr_convolution_step};
pub use less::{
    incr_convolution_les, incr_convolution_les_step, incr_convolution_less_adv,
    incr_convolution_lmix_rmix, incr_convolution_ret_less, incr_less_adv_buf, incr_lmix_rmix_buf,
};
pub use lmix::{conv_lmix_mat, conv_ret_lmix_kernel, incr_convolution_lmix};
pub use mat::{conv_mat, incr_convolution_mat};
pub use pseudo::{
    pseudo_convolution, pseudo_convolution_mat, pseudo_convolution_timestep, pseudo_incr_convolution,
    pseudo_incr_convolution_lmix, pseudo_incr_convolution_mat, pseudo_incr_convolution_step,
};
pub use ret::incr_convolution_ret;
pub use timestep::{convolution, convolution_ab, convolution_time_step};
pub use workspace::{ConvWorkspace, FOp};

pub(crate) use density_matrix::convolution_density_matrix_herm;
pub(crate) use less::{
    conv_less_adv_kernel, conv_lmix_rmix_kernel, conv_ret_less_kernel, incr_convolution_less_adv_step,
    incr_convolution_lmix_rmix_step, incr_les_diagcc,
};
pub(crate) use lmix::{conv_lmix_mat_loop, incr_convolution_lmix_step};
pub(crate) use mat::{cf_mat0, conv_mat_mat_1};
pub(crate) use pseudo::pseudo_incr_convolution_lmix_step_c;
pub(crate) use ret::{conv_ret_row, incr_convolution_ret_step};
pub(crate) use workspace::{af, adjoint_block, fb, mul_acc, mul_acc_c};
