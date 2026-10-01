//! Continuous batching: sequences join the running batch as cache pages
//! allow, every step decodes one token for each running sequence, and a
//! sequence leaves (freeing its pages) as soon as it finishes.

use super::argmax;
use super::engine::{Attn, Engine};
use std::time::Instant;

pub struct Request {
    pub prompt: Vec<i32>,
    pub max_new: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Output {
    pub tokens: Vec<i32>,
    /// Seconds from the start of `generate` to the first new token.
    pub first_token_s: f64,
    pub done_s: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub prefill_steps: usize,
    pub decode_steps: usize,
    pub prompt_tokens: usize,
    pub generated: usize,
    pub seconds: f64,
    /// Seconds in decode steps, and the tokens they produced.
    pub decode_seconds: f64,
    pub decode_tokens: usize,
    pub largest_batch: usize,
}

struct Seq {
    id: usize,
    pages: Vec<i32>,
    /// Tokens in the cache.
    len: usize,
    last: i32,
    left: usize,
}

/// Greedy generation for every request; stops a sequence at `max_new`
/// tokens or the model's end-of-sequence token.
pub fn generate(e: &mut Engine, reqs: &[Request], max_batch: usize) -> Result<(Vec<Output>, Stats), String> {
    generate_with(e, reqs, max_batch, true)
}

/// `generate`, choosing whether new prompts share prefill steps (`packed`)
/// or each takes its own.
pub fn generate_with(
    e: &mut Engine,
    reqs: &[Request],
    max_batch: usize,
    packed: bool,
) -> Result<(Vec<Output>, Stats), String> {
    let start = Instant::now();
    let p = e.lim.page;
    let mut free: Vec<i32> = (0..e.lim.pages as i32).rev().collect();
    let mut waiting: std::collections::VecDeque<usize> = (0..reqs.len()).collect();
    let mut running: Vec<Seq> = Vec::new();
    let mut out = vec![Output::default(); reqs.len()];
    let mut st = Stats::default();
    let max_batch = max_batch.min(e.lim.max_tokens);
    let eos = e.cfg.eos;
    while !waiting.is_empty() || !running.is_empty() {
        // Admit what fits, packing the new prompts into one prefill step:
        // as many as the step's token budget, the batch and the free cache
        // pages allow. Pages for a whole sequence are reserved up front.
        let mut admit: Vec<(usize, Vec<i32>)> = Vec::new();
        let mut tokens = 0;
        while let Some(&id) = waiting.front() {
            let r = &reqs[id];
            let need = (r.prompt.len() + r.max_new).div_ceil(p);
            if r.prompt.is_empty() || r.prompt.len() > e.lim.max_tokens || need > e.lim.max_seq_pages {
                return Err(format!(
                    "request {id}: prompt of {} tokens does not fit",
                    r.prompt.len()
                ));
            }
            let room = running.len() + admit.len() < max_batch && tokens + r.prompt.len() <= e.lim.max_tokens;
            if !room || need > free.len() || (!packed && !admit.is_empty()) {
                break;
            }
            waiting.pop_front();
            tokens += r.prompt.len();
            admit.push((id, (0..need).map(|_| free.pop().unwrap()).collect()));
        }
        if !admit.is_empty() {
            let (mut toks, mut pos, mut slots, mut starts, mut last) = (vec![], vec![], vec![], vec![], vec![]);
            for (id, pages) in &admit {
                let pr = &reqs[*id].prompt;
                let s0 = toks.len() as i32;
                for (i, &t) in pr.iter().enumerate() {
                    toks.push(t);
                    pos.push(i as i32);
                    slots.push(pages[i / p] * p as i32 + (i % p) as i32);
                    starts.push(s0);
                }
                last.push(toks.len() - 1);
            }
            let attn = if admit.len() == 1 {
                Attn::Prefill
            } else {
                Attn::Packed { starts: &starts }
            };
            let logits = e.step(&toks, &pos, &slots, attn, &last)?;
            st.prefill_steps += 1;
            st.prompt_tokens += toks.len();
            for ((id, pages), l) in admit.into_iter().zip(logits) {
                let t = argmax(&l) as i32;
                let n = reqs[id].prompt.len();
                out[id].tokens.push(t);
                out[id].first_token_s = start.elapsed().as_secs_f64();
                let s = Seq {
                    id,
                    pages,
                    len: n,
                    last: t,
                    left: reqs[id].max_new - 1,
                };
                if s.left == 0 || t == eos {
                    out[id].done_s = start.elapsed().as_secs_f64();
                    free.extend(&s.pages);
                } else {
                    running.push(s);
                }
            }
            continue;
        }
        if running.is_empty() {
            continue;
        }
        // One token for every running sequence.
        let t0 = Instant::now();
        let toks: Vec<i32> = running.iter().map(|s| s.last).collect();
        let pos: Vec<i32> = running.iter().map(|s| s.len as i32).collect();
        let slots: Vec<i32> = running
            .iter()
            .map(|s| s.pages[s.len / p] * p as i32 + (s.len % p) as i32)
            .collect();
        let tables: Vec<Vec<i32>> = running.iter().map(|s| s.pages.clone()).collect();
        let lens: Vec<i32> = running.iter().map(|s| s.len as i32 + 1).collect();
        let want: Vec<usize> = (0..running.len()).collect();
        let logits = e.step(
            &toks,
            &pos,
            &slots,
            Attn::Decode {
                tables: &tables,
                lens: &lens,
            },
            &want,
        )?;
        st.decode_steps += 1;
        st.largest_batch = st.largest_batch.max(running.len());
        st.decode_tokens += running.len();
        let mut keep = Vec::new();
        for (mut s, l) in running.drain(..).zip(logits) {
            let t = argmax(&l) as i32;
            s.len += 1;
            s.last = t;
            s.left -= 1;
            out[s.id].tokens.push(t);
            if s.left == 0 || t == eos {
                out[s.id].done_s = start.elapsed().as_secs_f64();
                free.extend(&s.pages);
            } else {
                keep.push(s);
            }
        }
        running = keep;
        st.decode_seconds += t0.elapsed().as_secs_f64();
    }
    st.generated = out.iter().map(|o| o.tokens.len()).sum();
    st.seconds = start.elapsed().as_secs_f64();
    Ok((out, st))
}
