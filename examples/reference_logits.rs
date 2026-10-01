//! The reference model's logits after a token sequence, for checking a
//! checkpoint's loading against another implementation.
//!
//! usage: cargo run --release --example reference_logits MODEL_DIR 1,2,3 > logits.json

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: reference_logits MODEL_DIR TOKEN_IDS");
        std::process::exit(2);
    }
    let w = tessel::llm::safetensors::load(std::path::Path::new(&a[1])).unwrap_or_else(|e| panic!("{e}"));
    let ids: Vec<i32> = a[2].split(',').map(|x| x.trim().parse().unwrap()).collect();
    eprintln!("{:?}", w.cfg);
    let t = std::time::Instant::now();
    let l = tessel::llm::reference::forward(&w, &ids).pop().unwrap();
    eprintln!("{} tokens in {:.1} s", ids.len(), t.elapsed().as_secs_f64());
    let v: Vec<String> = l.iter().map(|x| format!("{x:e}")).collect();
    println!("[{}]", v.join(", "));
}
