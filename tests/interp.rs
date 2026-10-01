//! The reference interpreter against plain implementations: the kernels'
//! semantics, before any backend is involved. Shapes are deliberately not
//! multiples of the tile sizes, so every edge mask is exercised.

mod common;

use common::*;
use tessel::ast::DType;
use tessel::interp::{Tensor, run};
use tessel::ir::{Spec, compile};

fn spec(shapes: &[&[usize]], meta: &[(&str, i64)]) -> Spec {
    Spec {
        shapes: shapes.iter().map(|s| s.to_vec()).collect(),
        meta: meta.iter().map(|(n, v)| (n.to_string(), *v)).collect(),
    }
}

#[test]
fn add() {
    let mut r = Rng::new(1);
    let n = 3000;
    let k = compile(&src("basic"), "add", &spec(&[&[n], &[n], &[n]], &[])).unwrap();
    let mut args = vec![
        r.tensor(DType::F32, &[n], 1.0),
        r.tensor(DType::F32, &[n], 1.0),
        Tensor::zeros(DType::F32, &[n]),
    ];
    run(&k, &mut args).unwrap();
    let want: Vec<f32> = args[0].data.iter().zip(&args[1].data).map(|(a, b)| a + b).collect();
    assert_eq!(args[2].data, want);
}

#[test]
fn softmax_and_rmsnorm() {
    let mut r = Rng::new(2);
    let (rows, cols) = (10, 700);
    let k = compile(&src("basic"), "softmax", &spec(&[&[rows, cols], &[rows, cols]], &[])).unwrap();
    let mut args = vec![
        r.tensor(DType::F32, &[rows, cols], 4.0),
        Tensor::zeros(DType::F32, &[rows, cols]),
    ];
    run(&k, &mut args).unwrap();
    assert_close("softmax", &args[1].data, &softmax_ref(&args[0].data, rows, cols), 1e-6);

    let k = compile(
        &src("basic"),
        "rmsnorm",
        &spec(&[&[rows, cols], &[cols], &[rows, cols]], &[]),
    )
    .unwrap();
    let mut args = vec![
        r.tensor(DType::F16, &[rows, cols], 2.0),
        r.tensor(DType::F16, &[cols], 1.0),
        Tensor::zeros(DType::F16, &[rows, cols]),
        Tensor::scalar(DType::F32, 1e-5),
    ];
    run(&k, &mut args).unwrap();
    let want = rmsnorm_ref(&args[0].data, &args[1].data, rows, cols, 1e-5);
    assert_close("rmsnorm", &args[2].data, &want, 2e-3);
}

#[test]
fn matmuls() {
    let mut r = Rng::new(3);
    let (m, kk, n) = (100, 70, 90);
    for (name, dt, tol) in [("matmul", DType::F16, 2e-3), ("matmul_f32", DType::F32, 1e-6)] {
        let k = compile(
            &src("matmul"),
            name,
            &spec(&[&[m, kk], &[kk, n], &[m, n]], &[("BK", 16)]),
        )
        .unwrap();
        let mut args = vec![
            r.tensor(dt, &[m, kk], 1.0),
            r.tensor(dt, &[kk, n], 1.0),
            Tensor::zeros(dt, &[m, n]),
        ];
        run(&k, &mut args).unwrap();
        let want = matmul_ref(&args[0].data, &args[1].data, m, kk, n, false);
        assert_close(name, &args[2].data, &want, tol);
    }
    let k = compile(
        &src("matmul"),
        "linear",
        &spec(&[&[m, kk], &[n, kk], &[n], &[m, n]], &[]),
    )
    .unwrap();
    let mut args = vec![
        r.tensor(DType::F16, &[m, kk], 1.0),
        r.tensor(DType::F16, &[n, kk], 1.0),
        r.tensor(DType::F16, &[n], 1.0),
        Tensor::zeros(DType::F16, &[m, n]),
    ];
    run(&k, &mut args).unwrap();
    let mut want = matmul_ref(&args[0].data, &args[1].data, m, kk, n, true);
    for (i, w) in want.iter_mut().enumerate() {
        *w += args[2].data[i % n];
    }
    assert_close("linear", &args[3].data, &want, 2e-3);
}

#[test]
fn flash_attention() {
    let mut r = Rng::new(4);
    let (h, s, d) = (2, 150, 32);
    let k = compile(
        &src("attention"),
        "flash_attention",
        &spec(&[&[h, s, d], &[h, s, d], &[h, s, d], &[h, s, d]], &[]),
    )
    .unwrap();
    let scale = 1.0 / (d as f32).sqrt();
    let mut args = vec![
        r.tensor(DType::F16, &[h, s, d], 1.0),
        r.tensor(DType::F16, &[h, s, d], 1.0),
        r.tensor(DType::F16, &[h, s, d], 1.0),
        Tensor::zeros(DType::F16, &[h, s, d]),
        Tensor::scalar(DType::F32, scale),
    ];
    run(&k, &mut args).unwrap();
    let want = attention_ref(&args[0].data, &args[1].data, &args[2].data, h, s, d, scale);
    assert_close("flash attention", &args[3].data, &want, 5e-3);
}

#[test]
fn paged_attention() {
    let mut r = Rng::new(5);
    let (b, h, d, p) = (3, 2, 32, 16);
    let c = paged_case(&mut r, b, h, d, p, &[37, 1, 64]);
    let shapes: Vec<Vec<usize>> = c.args[..6].iter().map(|t| t.shape.clone()).collect();
    let k = compile(&src("attention"), "paged_attention", &Spec { shapes, meta: vec![] }).unwrap();
    let mut args = c.args;
    run(&k, &mut args).unwrap();
    assert_close("paged attention", &args[5].data, &c.want, 5e-3);
}

#[test]
fn errors_name_the_line() {
    let e = compile(
        "kernel k(X: f32[N]):\n    grid(1)\n    x = X[0 : +3]\n",
        "k",
        &spec(&[&[8]], &[]),
    )
    .unwrap_err();
    assert!(e.contains("line 3") && e.contains("powers of two"), "{e}");
    let e = compile(
        "kernel k(X: f32[N]):\n    grid(1)\n    y = foo(1)\n",
        "k",
        &spec(&[&[8]], &[]),
    )
    .unwrap_err();
    assert!(e.contains("unknown function foo"), "{e}");
}
