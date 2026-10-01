fn main() {
    let a: Vec<String> = std::env::args().collect();
    let src = std::fs::read_to_string(&a[1]).unwrap();
    let nv = tessel::runtime::nvrtc().unwrap();
    match nv.compile(&src, a.get(2).map(|s| s.parse().unwrap()).unwrap_or(75), true) {
        Ok((_, log)) => println!(
            "{}",
            log.lines()
                .filter(|l| l.contains("spill") || l.contains("Used") || l.contains("stack"))
                .collect::<Vec<_>>()
                .join("\n")
        ),
        Err(e) => println!("{e}"),
    }
}
