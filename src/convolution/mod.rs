// Project : KadanoffBaym
// File    : src/convolution/mod.rs
// Purpose : real-time convolution kernels (blueprint: julia/src/convolution/)

pub mod bubble;
pub mod pseudo;
pub mod density_matrix;
pub mod less;
pub mod lmix;
pub mod mat;
pub mod ret;
pub mod incr;
pub mod response;
pub mod timestep;
pub mod workspace;

pub(crate) use response::{response_convolution, response_convolution_step};

pub use timestep::{convolution, convolution_ab, convolution_time_step};

pub use less::{
    incr_convolution_les, incr_convolution_les_step, incr_convolution_less_adv,
    incr_convolution_lmix_rmix, incr_convolution_ret_less,
};
pub(crate) use less::{conv_less_adv, conv_ret_less};
pub use lmix::{conv_lmix_mat, incr_convolution_lmix};
pub(crate) use lmix::{conv_lmix_mat_f0, incr_convolution_lmix_step};
pub use mat::{conv_mat, incr_convolution_mat};
pub(crate) use mat::{
    conv_mat_f0, conv_mat_mat_1, conv_mat_mat_1p, conv_mat_mat_2, conv_mat_mat_2p,
};
pub use ret::incr_convolution_ret;
pub(crate) use ret::{conv_ret, incr_convolution_ret_step};

pub use workspace::{ConvWorkspace, FOp};

pub(crate) use workspace::{af, adjoint_block, fb, mul_acc, mul_acc_c};
