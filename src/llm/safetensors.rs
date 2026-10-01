//! Reading Hugging Face checkpoints: config.json and safetensors files
//! (one, or several listed in model.safetensors.index.json), for Llama-
//! family models (Llama, TinyLlama, Mistral-style GQA without biases).

use super::{Config, Layer, Weights};
use crate::half::f32_to_f16;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// A JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn num(&self) -> Option<f64> {
        match self {
            Json::Num(x) => Some(*x),
            _ => None,
        }
    }
    pub fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
}

pub fn parse_json(s: &str) -> Result<Json, String> {
    let b = s.as_bytes();
    let mut i = 0;
    let v = value(b, &mut i)?;
    ws(b, &mut i);
    if i != b.len() {
        return Err(format!("json: trailing data at byte {i}"));
    }
    Ok(v)
}

fn ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && b[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

fn value(b: &[u8], i: &mut usize) -> Result<Json, String> {
    ws(b, i);
    let err = |i: usize| format!("json: unexpected input at byte {i}");
    match b.get(*i) {
        Some(b'{') => {
            *i += 1;
            let mut kv = Vec::new();
            ws(b, i);
            if b.get(*i) == Some(&b'}') {
                *i += 1;
                return Ok(Json::Obj(kv));
            }
            loop {
                ws(b, i);
                let Json::Str(k) = value(b, i)? else {
                    return Err(err(*i));
                };
                ws(b, i);
                if b.get(*i) != Some(&b':') {
                    return Err(err(*i));
                }
                *i += 1;
                kv.push((k, value(b, i)?));
                ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b'}') => {
                        *i += 1;
                        return Ok(Json::Obj(kv));
                    }
                    _ => return Err(err(*i)),
                }
            }
        }
        Some(b'[') => {
            *i += 1;
            let mut v = Vec::new();
            ws(b, i);
            if b.get(*i) == Some(&b']') {
                *i += 1;
                return Ok(Json::Arr(v));
            }
            loop {
                v.push(value(b, i)?);
                ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b']') => {
                        *i += 1;
                        return Ok(Json::Arr(v));
                    }
                    _ => return Err(err(*i)),
                }
            }
        }
        Some(b'"') => {
            *i += 1;
            let mut s = String::new();
            loop {
                match b.get(*i) {
                    None => return Err(err(*i)),
                    Some(b'"') => {
                        *i += 1;
                        return Ok(Json::Str(s));
                    }
                    Some(b'\\') => {
                        let c = *b.get(*i + 1).ok_or_else(|| err(*i))?;
                        *i += 2;
                        match c {
                            b'n' => s.push('\n'),
                            b't' => s.push('\t'),
                            b'r' => s.push('\r'),
                            b'b' => s.push('\u{8}'),
                            b'f' => s.push('\u{c}'),
                            b'u' => {
                                let hex = |at: usize| -> Result<u32, String> {
                                    let h = b.get(at..at + 4).ok_or_else(|| err(at))?;
                                    u32::from_str_radix(std::str::from_utf8(h).map_err(|_| err(at))?, 16)
                                        .map_err(|_| err(at))
                                };
                                let mut cp = hex(*i)?;
                                *i += 4;
                                // A surrogate pair.
                                if (0xd800..0xdc00).contains(&cp) && b.get(*i..*i + 2) == Some(b"\\u") {
                                    let lo = hex(*i + 2)?;
                                    cp = 0x10000 + ((cp - 0xd800) << 10) + (lo.wrapping_sub(0xdc00) & 0x3ff);
                                    *i += 6;
                                }
                                s.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            }
                            c => s.push(c as char),
                        }
                    }
                    Some(_) => {
                        // A run of plain UTF-8 bytes.
                        let st = *i;
                        while *i < b.len() && b[*i] != b'"' && b[*i] != b'\\' {
                            *i += 1;
                        }
                        s.push_str(std::str::from_utf8(&b[st..*i]).map_err(|_| err(st))?);
                    }
                }
            }
        }
        Some(b't') if b[*i..].starts_with(b"true") => {
            *i += 4;
            Ok(Json::Bool(true))
        }
        Some(b'f') if b[*i..].starts_with(b"false") => {
            *i += 5;
            Ok(Json::Bool(false))
        }
        Some(b'n') if b[*i..].starts_with(b"null") => {
            *i += 4;
            Ok(Json::Null)
        }
        Some(c) if *c == b'-' || c.is_ascii_digit() => {
            let st = *i;
            while *i < b.len() && (b[*i].is_ascii_digit() || b"+-.eE".contains(&b[*i])) {
                *i += 1;
            }
            let t = std::str::from_utf8(&b[st..*i]).unwrap();
            t.parse().map(Json::Num).map_err(|_| err(st))
        }
        _ => Err(err(*i)),
    }
}

/// The model's shape from its config.json.
pub fn config(dir: &Path) -> Result<Config, String> {
    let p = dir.join("config.json");
    let j = parse_json(&std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?)?;
    let n = |k: &str| {
        j.get(k)
            .and_then(Json::num)
            .ok_or_else(|| format!("config.json: no {k}"))
    };
    let n_or = |k: &str, d: f64| j.get(k).and_then(Json::num).unwrap_or(d);
    if let Some(s) = j.get("rope_scaling")
        && *s != Json::Null
    {
        return Err("config.json: rope_scaling is not supported".into());
    }
    let heads = n("num_attention_heads")? as usize;
    let dim = n("hidden_size")? as usize;
    let eos = match j.get("eos_token_id") {
        Some(Json::Num(x)) => *x as i32,
        Some(Json::Arr(v)) => v.first().and_then(Json::num).unwrap_or(-1.0) as i32,
        _ => -1,
    };
    Ok(Config {
        dim,
        layers: n("num_hidden_layers")? as usize,
        heads,
        kv_heads: n_or("num_key_value_heads", heads as f64) as usize,
        head_dim: n_or("head_dim", (dim / heads) as f64) as usize,
        ffn: n("intermediate_size")? as usize,
        vocab: n("vocab_size")? as usize,
        eps: n_or("rms_norm_eps", 1e-6) as f32,
        rope_theta: n_or("rope_theta", 10000.0) as f32,
        max_pos: (n_or("max_position_embeddings", 2048.0) as usize).min(8192),
        bos: n_or("bos_token_id", 1.0) as i32,
        eos,
    })
}

struct Entry {
    file: PathBuf,
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

/// Every tensor of the checkpoint's safetensors files, by name.
fn index(dir: &Path) -> Result<HashMap<String, Entry>, String> {
    let single = dir.join("model.safetensors");
    let files: Vec<PathBuf> = if single.exists() {
        vec![single]
    } else {
        let p = dir.join("model.safetensors.index.json");
        let j = parse_json(&std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?)?;
        let Some(Json::Obj(map)) = j.get("weight_map") else {
            return Err("index: no weight_map".into());
        };
        let mut f: Vec<PathBuf> = map.iter().filter_map(|(_, v)| v.str().map(|s| dir.join(s))).collect();
        f.sort();
        f.dedup();
        f
    };
    let mut out = HashMap::new();
    for file in files {
        let mut fh = File::open(&file).map_err(|e| format!("{}: {e}", file.display()))?;
        let mut len = [0u8; 8];
        fh.read_exact(&mut len).map_err(|e| e.to_string())?;
        let n = u64::from_le_bytes(len);
        let mut head = vec![0u8; n as usize];
        fh.read_exact(&mut head).map_err(|e| e.to_string())?;
        let j = parse_json(std::str::from_utf8(&head).map_err(|e| e.to_string())?)?;
        let Json::Obj(kv) = j else {
            return Err("safetensors: bad header".into());
        };
        for (name, v) in kv {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v.get("dtype").and_then(Json::str).unwrap_or("").to_string();
            let shape = match v.get("shape") {
                Some(Json::Arr(a)) => a.iter().map(|x| x.num().unwrap_or(0.0) as usize).collect(),
                _ => Vec::new(),
            };
            let (start, end) = match v.get("data_offsets") {
                Some(Json::Arr(a)) if a.len() == 2 => (
                    a[0].num().ok_or("bad offset")? as u64,
                    a[1].num().ok_or("bad offset")? as u64,
                ),
                _ => return Err(format!("{name}: no data_offsets")),
            };
            out.insert(
                name,
                Entry {
                    file: file.clone(),
                    dtype,
                    shape,
                    start: 8 + n + start,
                    end: 8 + n + end,
                },
            );
        }
    }
    Ok(out)
}

/// A tensor as f16 bits, checked against the expected shape.
fn read(ix: &HashMap<String, Entry>, name: &str, shape: &[usize]) -> Result<Vec<u16>, String> {
    let e = ix.get(name).ok_or_else(|| format!("checkpoint has no {name}"))?;
    if e.shape != shape {
        return Err(format!("{name}: shape {:?}, expected {shape:?}", e.shape));
    }
    let mut f = File::open(&e.file).map_err(|err| err.to_string())?;
    f.seek(SeekFrom::Start(e.start)).map_err(|err| err.to_string())?;
    let mut b = vec![0u8; (e.end - e.start) as usize];
    f.read_exact(&mut b).map_err(|err| err.to_string())?;
    let n: usize = shape.iter().product();
    Ok(match e.dtype.as_str() {
        "F16" if b.len() == 2 * n => b.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect(),
        "BF16" if b.len() == 2 * n => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32_to_f16(f32::from_bits((u16::from_le_bytes(*c) as u32) << 16)))
            .collect(),
        "F32" if b.len() == 4 * n => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32_to_f16(f32::from_le_bytes(*c)))
            .collect(),
        d => return Err(format!("{name}: dtype {d} not supported")),
    })
}

/// A Llama-family checkpoint directory (config.json and safetensors).
pub fn load(dir: &Path) -> Result<Weights, String> {
    let cfg = config(dir)?;
    let ix = index(dir)?;
    if let Some(b) = ix.keys().find(|k| k.ends_with(".bias")) {
        return Err(format!("{b}: models with biases are not supported"));
    }
    let (dm, hd) = (cfg.dim, cfg.head_dim);
    let mut layers = Vec::new();
    for i in 0..cfg.layers {
        let p = format!("model.layers.{i}.");
        let mut wqkv = read(&ix, &format!("{p}self_attn.q_proj.weight"), &[cfg.heads * hd, dm])?;
        wqkv.extend(read(
            &ix,
            &format!("{p}self_attn.k_proj.weight"),
            &[cfg.kv_heads * hd, dm],
        )?);
        wqkv.extend(read(
            &ix,
            &format!("{p}self_attn.v_proj.weight"),
            &[cfg.kv_heads * hd, dm],
        )?);
        layers.push(Layer {
            attn_norm: read(&ix, &format!("{p}input_layernorm.weight"), &[dm])?,
            wqkv,
            wo: read(&ix, &format!("{p}self_attn.o_proj.weight"), &[dm, cfg.heads * hd])?,
            mlp_norm: read(&ix, &format!("{p}post_attention_layernorm.weight"), &[dm])?,
            wg: read(&ix, &format!("{p}mlp.gate_proj.weight"), &[cfg.ffn, dm])?,
            wu: read(&ix, &format!("{p}mlp.up_proj.weight"), &[cfg.ffn, dm])?,
            wd: read(&ix, &format!("{p}mlp.down_proj.weight"), &[dm, cfg.ffn])?,
        });
    }
    let lm_head = if ix.contains_key("lm_head.weight") {
        Some(read(&ix, "lm_head.weight", &[cfg.vocab, dm])?)
    } else {
        None
    };
    Ok(Weights {
        embed: read(&ix, "model.embed_tokens.weight", &[cfg.vocab, dm])?,
        norm: read(&ix, "model.norm.weight", &[dm])?,
        layers,
        lm_head,
        cfg,
    })
}

/// Writes `w` as a checkpoint directory (F16), laid out as Hugging Face
/// lays one out: for tests, and for trying the engine without a download.
pub fn save(w: &Weights, dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let c = &w.cfg;
    let cfg = format!(
        "{{\"model_type\": \"llama\", \"hidden_size\": {}, \"num_hidden_layers\": {}, \"num_attention_heads\": {}, \"num_key_value_heads\": {}, \"head_dim\": {}, \"intermediate_size\": {}, \"vocab_size\": {}, \"rms_norm_eps\": {:e}, \"rope_theta\": {:.1}, \"max_position_embeddings\": {}, \"bos_token_id\": {}, \"eos_token_id\": {}, \"rope_scaling\": null, \"tie_word_embeddings\": {}}}",
        c.dim,
        c.layers,
        c.heads,
        c.kv_heads,
        c.head_dim,
        c.ffn,
        c.vocab,
        c.eps,
        c.rope_theta,
        c.max_pos,
        c.bos,
        c.eos,
        w.lm_head.is_none()
    );
    std::fs::write(dir.join("config.json"), cfg).map_err(|e| e.to_string())?;
    let (dm, hd) = (c.dim, c.head_dim);
    let mut tensors: Vec<(String, Vec<usize>, &[u16])> = vec![
        ("model.embed_tokens.weight".into(), vec![c.vocab, dm], &w.embed),
        ("model.norm.weight".into(), vec![dm], &w.norm),
    ];
    if let Some(h) = &w.lm_head {
        tensors.push(("lm_head.weight".into(), vec![c.vocab, dm], h));
    }
    let (q, k) = (c.heads * hd * dm, c.kv_heads * hd * dm);
    for (i, l) in w.layers.iter().enumerate() {
        let p = format!("model.layers.{i}.");
        tensors.push((format!("{p}input_layernorm.weight"), vec![dm], &l.attn_norm));
        tensors.push((
            format!("{p}self_attn.q_proj.weight"),
            vec![c.heads * hd, dm],
            &l.wqkv[..q],
        ));
        tensors.push((
            format!("{p}self_attn.k_proj.weight"),
            vec![c.kv_heads * hd, dm],
            &l.wqkv[q..q + k],
        ));
        tensors.push((
            format!("{p}self_attn.v_proj.weight"),
            vec![c.kv_heads * hd, dm],
            &l.wqkv[q + k..],
        ));
        tensors.push((format!("{p}self_attn.o_proj.weight"), vec![dm, c.heads * hd], &l.wo));
        tensors.push((format!("{p}post_attention_layernorm.weight"), vec![dm], &l.mlp_norm));
        tensors.push((format!("{p}mlp.gate_proj.weight"), vec![c.ffn, dm], &l.wg));
        tensors.push((format!("{p}mlp.up_proj.weight"), vec![c.ffn, dm], &l.wu));
        tensors.push((format!("{p}mlp.down_proj.weight"), vec![dm, c.ffn], &l.wd));
    }
    let mut head = Vec::new();
    let mut off = 0usize;
    for (name, shape, data) in &tensors {
        let n = data.len() * 2;
        head.push(format!(
            "\"{name}\": {{\"dtype\": \"F16\", \"shape\": {shape:?}, \"data_offsets\": [{off}, {}]}}",
            off + n
        ));
        off += n;
    }
    let mut h = format!("{{{}}}", head.join(", ")).into_bytes();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for (_, _, data) in &tensors {
        out.extend(data.iter().flat_map(|x| x.to_le_bytes()));
    }
    std::fs::write(dir.join("model.safetensors"), out).map_err(|e| e.to_string())
}
