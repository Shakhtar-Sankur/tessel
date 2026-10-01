use tessel::ir::{Spec, compile};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let src = std::fs::read_to_string(&a[1]).unwrap();
    let shapes: Vec<Vec<usize>> = a[3]
        .split(';')
        .map(|s| s.split('x').map(|d| d.parse().unwrap()).collect())
        .collect();
    let meta: Vec<(String, i64)> = a
        .get(4)
        .map(|m| {
            m.split(',')
                .filter(|s| !s.is_empty())
                .map(|kv| {
                    let (k, v) = kv.split_once('=').unwrap();
                    (k.to_string(), v.parse().unwrap())
                })
                .collect()
        })
        .unwrap_or_default();
    let k = compile(&src, &a[2], &Spec { shapes, meta }).unwrap();
    let g = tessel::cuda::generate(&k, &tessel::cuda::Options::default()).unwrap();
    let body = &g.source[g.source.rfind("KGLOBAL(").unwrap()..];
    println!("{body}");
    eprintln!("smem {} threads {} grid {:?}", g.smem, g.threads, g.grid);
}
