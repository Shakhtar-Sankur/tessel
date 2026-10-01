//! The engine (every step a sequence of tessel kernels, on the emulator)
//! against the f32 reference model, on a small random Llama: logits after
//! a prompt, and every token of a continuous-batched generation.

use tessel::llm::engine::{Attn, Engine, Limits};
use tessel::llm::sched::{Request, generate};
use tessel::llm::{Config, Weights, argmax, reference};
use tessel::runtime::Device;

fn limits() -> Limits {
    Limits {
        page: 8,
        pages: 32,
        max_tokens: 16,
        max_seq_pages: 8,
    }
}

/// Largest difference relative to the largest reference value.
fn rel(got: &[f32], want: &[f32]) -> f32 {
    let d = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    d / want.iter().map(|x| x.abs()).fold(0f32, f32::max)
}

#[test]
fn prefill_and_batched_decode_match_the_reference() {
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 7);
    let mut e = Engine::new(Device::Emu, &w, limits()).unwrap();

    // One prompt: the logits after its last token.
    let prompt: Vec<i32> = vec![1, 17, 200, 33, 5, 99, 64, 3, 250, 12];
    let n = prompt.len();
    let pos: Vec<i32> = (0..n as i32).collect();
    let slots: Vec<i32> = (0..n as i32).collect();
    let got = e.step(&prompt, &pos, &slots, Attn::Prefill, &[n - 1]).unwrap();
    let want = reference::forward(&w, &prompt);
    let err = rel(&got[0], &want[n - 1]);
    eprintln!("prefill: logits within {err:.2e} of the reference (relative to the largest)");
    assert!(err < 2e-2, "prefill logits differ by {err}");

    // Three requests of different lengths, decoded together.
    let reqs = vec![
        Request {
            prompt: vec![1, 5, 9],
            max_new: 6,
        },
        Request {
            prompt: prompt.clone(),
            max_new: 4,
        },
        Request {
            prompt: vec![1, 100, 101, 102, 103, 104, 105],
            max_new: 9,
        },
    ];
    let (outs, st) = generate(&mut e, &reqs, 3).unwrap();
    assert_eq!(st.largest_batch, 3);
    let mut exact = 0;
    for (r, o) in reqs.iter().zip(&outs) {
        assert_eq!(o.tokens.len(), r.max_new);
        // Each token the engine picked is the reference's choice, or within
        // rounding of it, given the same history.
        let mut seq = r.prompt.clone();
        for &t in &o.tokens {
            let l = reference::forward(&w, &seq).pop().unwrap();
            let best = argmax(&l);
            let scale = l.iter().map(|x| x.abs()).fold(0f32, f32::max);
            assert!(
                t as usize == best || l[best] - l[t as usize] < 2e-2 * scale,
                "token {t} where the reference picks {best} ({} vs {})",
                l[t as usize],
                l[best]
            );
            exact += (t as usize == best) as usize;
            seq.push(t);
        }
    }
    eprintln!(
        "generation: {exact} of {} tokens the reference's own choice; {st:?}",
        st.generated
    );
}

#[test]
fn checkpoints_round_trip() {
    use tessel::llm::safetensors::{load, parse_json, save};
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 3);
    let dir = std::env::temp_dir().join(format!("tessel-ckpt-{}", std::process::id()));
    save(&w, &dir).unwrap();
    let r = load(&dir).unwrap();
    assert_eq!(r.cfg, cfg);
    assert_eq!(r.embed, w.embed);
    assert_eq!(r.lm_head, w.lm_head);
    for (a, b) in r.layers.iter().zip(&w.layers) {
        assert!(a.wqkv == b.wqkv && a.wo == b.wo && a.wg == b.wg && a.wu == b.wu && a.wd == b.wd);
    }
    std::fs::remove_dir_all(&dir).unwrap();
    let j = parse_json(r#"{"a": [1, -2.5e3, true, null], "b": "x\"é😀y"}"#).unwrap();
    assert_eq!(j.get("b").unwrap().str(), Some("x\"é😀y"));
}

/// Every tile configuration the engine's GPU tuner may pick, for each of
/// its matmul kernels, at a decode size and a prompt size: the emulator
/// against the interpreter.
#[test]
fn every_tuning_candidate_is_correct() {
    use tessel::ast::DType;
    use tessel::bench::Rng;
    use tessel::cuda::Options;
    use tessel::interp::{self, Tensor};
    use tessel::ir::{Spec, compile};
    use tessel::llm::engine::mm_candidates;
    let src = tessel::kernels::source("llm").unwrap();
    let mut r = Rng::new(5);
    let (k, n) = (256usize, 96usize);
    for m in [16usize, 64] {
        let cases: Vec<(&str, Vec<Tensor>, bool)> = vec![
            (
                "linear",
                vec![
                    r.tensor(DType::F16, &[m, k], 1.0),
                    r.tensor(DType::F16, &[n, k], 0.1),
                    Tensor::zeros(DType::F16, &[m, n]),
                ],
                false,
            ),
            (
                "linear_residual",
                vec![
                    r.tensor(DType::F16, &[m, k], 1.0),
                    r.tensor(DType::F16, &[n, k], 0.1),
                    r.tensor(DType::F32, &[m, n], 1.0),
                ],
                false,
            ),
            (
                "logits",
                vec![
                    r.tensor(DType::F16, &[m, k], 1.0),
                    r.tensor(DType::F16, &[n, k], 0.1),
                    Tensor::zeros(DType::F32, &[m, n]),
                ],
                false,
            ),
            (
                "gate_up",
                vec![
                    r.tensor(DType::F16, &[m, k], 1.0),
                    r.tensor(DType::F16, &[n, k], 0.1),
                    r.tensor(DType::F16, &[n, k], 0.1),
                    Tensor::zeros(DType::F16, &[m, n]),
                ],
                true,
            ),
        ];
        for (name, args, two) in cases {
            let shapes: Vec<Vec<usize>> = args.iter().map(|t| t.shape.clone()).collect();
            for (meta, warps) in mm_candidates(m, two) {
                let spec = Spec {
                    shapes: shapes.clone(),
                    meta: meta.iter().map(|(a, b)| (a.to_string(), *b)).collect(),
                };
                let kern = compile(src, name, &spec).unwrap();
                let mut want = args.clone();
                interp::run(&kern, &mut want).unwrap();
                let c = tessel::runtime::compile(&kern, tessel::runtime::Device::Emu, &Options { warps, arch: 75 })
                    .unwrap_or_else(|e| panic!("{name} {meta:?} w{warps}: {e}"));
                let mut got = args.clone();
                c.run(&mut got).unwrap();
                let out = got.len() - 1;
                let scale = want[out].data.iter().map(|x| x.abs()).fold(0f32, f32::max);
                let err = got[out]
                    .data
                    .iter()
                    .zip(&want[out].data)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    err <= 1e-2 * scale,
                    "{name} m={m} {meta:?} w{warps}: error {err} of {scale}"
                );
            }
        }
    }
}
