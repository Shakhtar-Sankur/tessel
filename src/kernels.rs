//! The kernel library, built into the binary.

pub const FILES: &[(&str, &str)] = &[
    ("basic", include_str!("../kernels/basic.tl")),
    ("matmul", include_str!("../kernels/matmul.tl")),
    ("attention", include_str!("../kernels/attention.tl")),
    ("llm", include_str!("../kernels/llm.tl")),
];

/// The source of kernel file `name` (without `.tl`).
pub fn source(name: &str) -> Option<&'static str> {
    FILES.iter().find(|(n, _)| *n == name).map(|f| f.1)
}
