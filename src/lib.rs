//! tessel: a tile language for GPU kernels. Kernels are written once,
//! against tiles (small blocks of values), and compiled for each target:
//! CUDA (NVIDIA tensor cores), Metal (Apple GPUs) and Pallas (Google TPUs).

pub mod ast;
pub mod bench;
pub mod cuda;
pub mod driver;
pub mod half;
pub mod interp;
pub mod ir;
pub mod jit;
pub mod kernels;
pub mod layout;
pub mod lexer;
pub mod runtime;
