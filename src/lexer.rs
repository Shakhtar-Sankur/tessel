//! Tokens of the tile language. Indentation is significant, as in Python:
//! the lexer turns changes of indentation into `Indent` and `Dedent`
//! tokens, and ends every logical line with `Newline`. Inside brackets,
//! line breaks are ignored.

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Name(String),
    Int(i64),
    Float(f64),
    Sym(&'static str),
    Newline,
    Indent,
    Dedent,
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub line: usize,
}

const SYMS: [&str; 26] = [
    "//", "**", "<=", ">=", "==", "!=", "+=", "-=", "*=", "(", ")", "[", "]", ",", ":", "+", "-", "*", "/", "%", "<",
    ">", "=", ".", "&", "|",
];

pub fn lex(src: &str) -> Result<Vec<Token>, String> {
    let mut out = Vec::new();
    let mut indents = vec![0usize];
    let mut depth = 0i32;
    for (n, raw) in src.lines().enumerate() {
        let line = n + 1;
        let code = match raw.find('#') {
            Some(i) => &raw[..i],
            None => raw,
        };
        if code.trim().is_empty() {
            continue;
        }
        if depth == 0 {
            if code.starts_with('\t') {
                return Err(format!("line {line}: indent with spaces, not tabs"));
            }
            let ind = code.len() - code.trim_start().len();
            let top = *indents.last().unwrap();
            if ind > top {
                indents.push(ind);
                out.push(Token { tok: Tok::Indent, line });
            } else {
                while ind < *indents.last().unwrap() {
                    indents.pop();
                    out.push(Token { tok: Tok::Dedent, line });
                }
                if ind != *indents.last().unwrap() {
                    return Err(format!("line {line}: indentation does not match any outer block"));
                }
            }
        }
        let b = code.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let c = b[i] as char;
            if c.is_whitespace() {
                i += 1;
                continue;
            }
            if c.is_ascii_alphabetic() || c == '_' {
                let s = i;
                while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                out.push(Token {
                    tok: Tok::Name(code[s..i].to_string()),
                    line,
                });
                continue;
            }
            if c.is_ascii_digit() || (c == '.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
                let s = i;
                let mut float = false;
                while i < b.len() {
                    let d = b[i] as char;
                    if d.is_ascii_digit() || d == '_' {
                        i += 1;
                    } else if d == '.' && !float {
                        float = true;
                        i += 1;
                    } else if (d == 'e' || d == 'E')
                        && i + 1 < b.len()
                        && ((b[i + 1] as char).is_ascii_digit() || b[i + 1] == b'-' || b[i + 1] == b'+')
                    {
                        float = true;
                        i += 2;
                    } else {
                        break;
                    }
                }
                let t = code[s..i].replace('_', "");
                let tok = if float {
                    Tok::Float(t.parse().map_err(|_| format!("line {line}: bad number {t}"))?)
                } else {
                    Tok::Int(t.parse().map_err(|_| format!("line {line}: bad number {t}"))?)
                };
                out.push(Token { tok, line });
                continue;
            }
            let rest = &code[i..];
            match SYMS.iter().find(|s| rest.starts_with(**s)) {
                Some(s) => {
                    match *s {
                        "(" | "[" => depth += 1,
                        ")" | "]" => depth -= 1,
                        _ => {}
                    }
                    out.push(Token { tok: Tok::Sym(s), line });
                    i += s.len();
                }
                None => return Err(format!("line {line}: unexpected character {c:?}")),
            }
        }
        if depth == 0 {
            out.push(Token {
                tok: Tok::Newline,
                line,
            });
        }
    }
    let last = src.lines().count();
    while indents.len() > 1 {
        indents.pop();
        out.push(Token {
            tok: Tok::Dedent,
            line: last,
        });
    }
    out.push(Token {
        tok: Tok::Eof,
        line: last,
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indentation_and_brackets() {
        let t = lex("kernel k(a: f32[N]):\n    x = f(1,\n          2.5)\n    y = x\n").unwrap();
        let toks: Vec<Tok> = t.into_iter().map(|t| t.tok).collect();
        assert_eq!(toks.iter().filter(|t| **t == Tok::Indent).count(), 1);
        assert_eq!(toks.iter().filter(|t| **t == Tok::Dedent).count(), 1);
        // The call spanning two lines is one logical line.
        assert_eq!(toks.iter().filter(|t| **t == Tok::Newline).count(), 3);
        assert!(toks.contains(&Tok::Float(2.5)));
    }
}
