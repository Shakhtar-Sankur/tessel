//! Generated CUDA, run on the emulator of the CUDA execution model (and on
//! a GPU when there is one), against the reference interpreter: every
//! kernel, under several tile shapes and warp counts.

mod common;

use common::*;
use tessel::ast::DType;
use tessel::cuda::{METAL, Options};
use tessel::interp::{Tensor, run};
use tessel::ir::{Spec, compile};
use tessel::runtime::{self, Device};

/// Where each kernel runs: the emulator with the CUDA code generator's
/// choices and with the Metal ones (no tensor cores; the same kernel body
/// as the Metal source), and the GPU when there is one.
fn targets() -> Vec<(Device, u32)> {
    let mut d = vec![(Device::Emu, 75), (Device::Emu, METAL)];
    if runtime::cuda().is_ok() {
        d.push((Device::Cuda, 75));
    }
    d
}

/// Runs `name` from `file` on every device with each meta/warps setting,
/// comparing every output array with the interpreter's.
fn check(file: &str, name: &str, args: &[Tensor], metas: &[(&[(&str, i64)], usize)], tol: f32) {
    let shapes: Vec<Vec<usize>> = args
        .iter()
        .filter(|t| !t.shape.is_empty())
        .map(|t| t.shape.clone())
        .collect();
    for (meta, warps) in metas {
        let spec = Spec {
            shapes: shapes.clone(),
            meta: meta.iter().map(|(n, v)| (n.to_string(), *v)).collect(),
        };
        let k = compile(&src(file), name, &spec).unwrap();
        let mut want = args.to_vec();
        run(&k, &mut want).unwrap();
        for (dev, arch) in targets() {
            let c = match runtime::compile(&k, dev, &Options { warps: *warps, arch }) {
                Ok(c) => c,
                // Metal gives a threadgroup 32 KB: larger tiles are refused.
                Err(e) if arch == METAL && e.contains("threadgroup memory") => continue,
                Err(e) => panic!("{name} {meta:?} warps {warps}: {e}"),
            };
            if arch == METAL {
                let m = c.code.metal.as_ref().expect("Metal source");
                assert!(m.contains("kernel void") && !m.contains("mma") && !m.contains("long long"));
            }
            let mut got = args.to_vec();
            c.run(&mut got).unwrap_or_else(|e| panic!("{name}: {e}"));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                if !g.shape.is_empty() {
                    assert_close(
                        &format!("{name} {meta:?} warps {warps} {dev:?} arch {arch} arg {i}"),
                        &g.data,
                        &w.data,
                        tol,
                    );
                }
            }
        }
    }
}

#[test]
fn elementwise_and_rows() {
    let mut r = Rng::new(11);
    let n = 3000;
    let args = vec![
        r.tensor(DType::F32, &[n], 1.0),
        r.tensor(DType::F32, &[n], 1.0),
        Tensor::zeros(DType::F32, &[n]),
    ];
    check(
        "basic",
        "add",
        &args,
        &[(&[], 4), (&[("B", 256)], 2), (&[("B", 64)], 8)],
        0.0,
    );

    let (rows, cols) = (10, 700);
    let args = vec![
        r.tensor(DType::F32, &[rows, cols], 4.0),
        Tensor::zeros(DType::F32, &[rows, cols]),
    ];
    check(
        "basic",
        "softmax",
        &args,
        &[(&[], 4), (&[("BR", 1)], 4), (&[("BR", 8)], 8), (&[("BR", 2)], 1)],
        1e-6,
    );

    let args = vec![
        r.tensor(DType::F16, &[rows, cols], 2.0),
        r.tensor(DType::F16, &[cols], 1.0),
        Tensor::zeros(DType::F16, &[rows, cols]),
        Tensor::scalar(DType::F32, 1e-5),
    ];
    check("basic", "rmsnorm", &args, &[(&[], 4), (&[("BR", 1)], 2)], 2e-3);
}

#[test]
fn matmuls() {
    let mut r = Rng::new(12);
    let (m, k, n) = (100, 70, 90);
    let args = vec![
        r.tensor(DType::F16, &[m, k], 1.0),
        r.tensor(DType::F16, &[k, n], 1.0),
        Tensor::zeros(DType::F16, &[m, n]),
    ];
    check(
        "matmul",
        "matmul",
        &args,
        &[
            (&[("BM", 64), ("BN", 64), ("BK", 32)], 4),
            (&[("BM", 32), ("BN", 64), ("BK", 16)], 2),
            (&[("BM", 128), ("BN", 32), ("BK", 32)], 8),
            (&[("BM", 16), ("BN", 16), ("BK", 16)], 1),
        ],
        2e-3,
    );
    let args = vec![
        r.tensor(DType::F32, &[m, k], 1.0),
        r.tensor(DType::F32, &[k, n], 1.0),
        Tensor::zeros(DType::F32, &[m, n]),
    ];
    check(
        "matmul",
        "matmul_f32",
        &args,
        &[(&[], 4), (&[("BM", 32), ("BN", 16), ("BK", 16)], 2)],
        1e-5,
    );
    let args = vec![
        r.tensor(DType::F16, &[m, k], 1.0),
        r.tensor(DType::F16, &[n, k], 1.0),
        r.tensor(DType::F16, &[n], 1.0),
        Tensor::zeros(DType::F16, &[m, n]),
    ];
    check(
        "matmul",
        "linear",
        &args,
        &[(&[("BM", 64), ("BN", 64), ("BK", 32)], 4)],
        2e-3,
    );
}

#[test]
fn attention() {
    let mut r = Rng::new(13);
    let (h, s, d) = (2, 100, 32);
    let args = vec![
        r.tensor(DType::F16, &[h, s, d], 1.0),
        r.tensor(DType::F16, &[h, s, d], 1.0),
        r.tensor(DType::F16, &[h, s, d], 1.0),
        Tensor::zeros(DType::F16, &[h, s, d]),
        Tensor::scalar(DType::F32, 1.0 / (d as f32).sqrt()),
    ];
    check(
        "attention",
        "flash_attention",
        &args,
        &[(&[("BM", 64), ("BN", 32)], 4), (&[("BM", 32), ("BN", 64)], 2)],
        3e-3,
    );

    let c = paged_case(&mut r, 3, 2, 32, 16, &[37, 1, 64]);
    check("attention", "paged_attention", &c.args, &[(&[], 4), (&[], 1)], 3e-3);
}
