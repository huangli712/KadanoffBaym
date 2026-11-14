//
// Handwritten TOML subset parser (D3: zero third-party dependencies).
//
// The grammar covers the syntax actually used by the tutor
// configurations: comments, `[table]` headers (including nested ones
// like `[drive.name]` and `[profile.release.drive.name]`), scalars
// (integers, floats, booleans, basic strings), same-type scalar arrays,
// and inline tables. Anything else is an error.
//

use std::collections::BTreeMap;

/// A parsed TOML value.
#[derive(Clone, Debug, PartialEq)]
pub enum TomlValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Array(Vec<TomlValue>),
    Table(BTreeMap<String, TomlValue>),
}

impl TomlValue {
    pub fn as_table(&self) -> Option<&BTreeMap<String, TomlValue>> {
        match self {
            TomlValue::Table(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            TomlValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            TomlValue::Float(x) => Some(*x),
            TomlValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            TomlValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            TomlValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn is_real(&self) -> bool {
        matches!(self, TomlValue::Float(_) | TomlValue::Int(_))
    }
}

/// Parse error (`line: message`).
#[derive(Clone, Debug, PartialEq)]
pub struct TomlError {
    pub line: usize,
    pub msg: String,
}

impl std::fmt::Display for TomlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "toml: line {}: {}", self.line, self.msg)
    }
}

impl std::error::Error for TomlError {}

fn err(line: usize, msg: impl Into<String>) -> TomlError {
    TomlError {
        line,
        msg: msg.into(),
    }
}

/// Split a line into the payload and a trailing comment, respecting
/// double-quoted strings.
fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut in_string = false;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '"' {
            in_string = !in_string;
        } else if c == '#' && !in_string {
            return &line[..i];
        }
        i += 1;
    }
    line
}

/// Parse a scalar token (no arrays/tables).
fn parse_scalar(tok: &str, line: usize) -> Result<TomlValue, TomlError> {
    let t = tok.trim();
    if t.is_empty() {
        return Err(err(line, "empty value"));
    }
    if t.starts_with('"') {
        if t.len() < 2 || !t.ends_with('"') {
            return Err(err(line, "unterminated string"));
        }
        return Ok(TomlValue::Str(t[1..t.len() - 1].to_string()));
    }
    if t == "true" {
        return Ok(TomlValue::Bool(true));
    }
    if t == "false" {
        return Ok(TomlValue::Bool(false));
    }
    // integer or float (also inf/nan, which the config layer rejects)
    let looks_numeric = t
        .strip_prefix(['+', '-'])
        .unwrap_or(t)
        .chars()
        .next()
        .map(|c| c.is_ascii_digit() || c == '.')
        .unwrap_or(false)
        || t == "inf"
        || t == "nan";
    if !looks_numeric {
        return Err(err(line, format!("unsupported value `{t}`")));
    }
    let is_float = t.contains('.') || t.contains('e') || t.contains('E') || t == "inf"
        || t == "nan"
        || t.starts_with("-inf")
        || t.starts_with("+inf");
    if is_float {
        let v: f64 = t
            .parse()
            .map_err(|_| err(line, format!("bad float `{t}`")))?;
        return Ok(TomlValue::Float(v));
    }
    let v: i64 = t
        .parse()
        .map_err(|_| err(line, format!("bad integer `{t}`")))?;
    Ok(TomlValue::Int(v))
}

/// Split a bracketed/braced body at the top-level commas.
fn split_top(s: &str, line: usize) -> Result<Vec<String>, TomlError> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '"' => {
                in_string = !in_string;
                cur.push(ch);
            }
            '[' | '{' if !in_string => {
                depth += 1;
                cur.push(ch);
            }
            ']' | '}' if !in_string => {
                if depth == 0 {
                    return Err(err(line, "unbalanced brackets"));
                }
                depth -= 1;
                cur.push(ch);
            }
            ',' if !in_string && depth == 0 => {
                out.push(std::mem::take(&mut cur));
            }
            _ => cur.push(ch),
        }
    }
    if depth != 0 {
        return Err(err(line, "unbalanced brackets"));
    }
    if !cur.trim().is_empty() || !out.is_empty() {
        out.push(cur);
    }
    Ok(out)
}

/// Parse a value token (scalar, array, or inline table).
fn parse_value(tok: &str, line: usize) -> Result<TomlValue, TomlError> {
    let t = tok.trim();
    if t.starts_with('[') {
        if !t.ends_with(']') {
            return Err(err(line, "unterminated array"));
        }
        let inner = &t[1..t.len() - 1];
        let mut items = Vec::new();
        for part in split_top(inner, line)? {
            let item = parse_scalar(&part, line)?;
            items.push(item);
        }
        // same-type scalar arrays only
        let mut kind: Option<&'static str> = None;
        for it in &items {
            let k = match it {
                TomlValue::Str(_) => "string",
                TomlValue::Int(_) | TomlValue::Float(_) => "real",
                TomlValue::Bool(_) => "bool",
                _ => "other",
            };
            if let Some(k0) = kind {
                if k0 != k {
                    return Err(err(line, "mixed-type arrays are not supported"));
                }
            }
            kind = Some(k);
        }
        return Ok(TomlValue::Array(items));
    }
    if t.starts_with('{') {
        if !t.ends_with('}') {
            return Err(err(line, "unterminated inline table"));
        }
        let inner = &t[1..t.len() - 1];
        let mut table = BTreeMap::new();
        for part in split_top(inner, line)? {
            let (k, v) = part
                .split_once('=')
                .ok_or_else(|| err(line, "expected `key = value` in the inline table"))?;
            let key = k.trim().to_string();
            if key.is_empty() {
                return Err(err(line, "empty key in the inline table"));
            }
            table.insert(key, parse_value(v, line)?);
        }
        return Ok(TomlValue::Table(table));
    }
    parse_scalar(t, line)
}

/// Descend (creating) the nested tables of a dotted path inside `root`.
fn descend<'a>(
    root: &'a mut BTreeMap<String, TomlValue>,
    path: &[&str],
    line: usize,
) -> Result<&'a mut BTreeMap<String, TomlValue>, TomlError> {
    let mut cur = root;
    for key in path {
        let entry = cur
            .entry(key.to_string())
            .or_insert_with(|| TomlValue::Table(BTreeMap::new()));
        match entry {
            TomlValue::Table(t) => cur = t,
            _ => return Err(err(line, format!("`{key}` is not a table"))),
        }
    }
    Ok(cur)
}

/// Parse a whole TOML document into a table.
pub fn parse_toml(text: &str) -> Result<BTreeMap<String, TomlValue>, TomlError> {
    let mut root: BTreeMap<String, TomlValue> = BTreeMap::new();
    // the table receiving the following key/value pairs
    let mut prefix: Vec<String> = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            if !line.ends_with(']') {
                return Err(err(line_no, "unterminated table header"));
            }
            let header = line[1..line.len() - 1].trim();
            if header.is_empty() {
                return Err(err(line_no, "empty table header"));
            }
            let parts: Vec<&str> = header.split('.').map(|s| s.trim()).collect();
            if parts.iter().any(|s| s.is_empty()) {
                return Err(err(line_no, "empty key in the table header"));
            }
            descend(&mut root, &parts, line_no)?;
            prefix = parts.into_iter().map(|s| s.to_string()).collect();
            continue;
        }
        let (k, v) = line
            .split_once('=')
            .ok_or_else(|| err(line_no, "expected `key = value`"))?;
        let key = k.trim().to_string();
        if key.is_empty() {
            return Err(err(line_no, "empty key"));
        }
        let value = parse_value(v, line_no)?;
        // dotted keys inside the current table (a.b = 1)
        let key_parts: Vec<&str> = key.split('.').collect();
        let (last, parents) = key_parts.split_last().unwrap();
        let mut path: Vec<&str> = prefix.iter().map(|s| s.as_str()).collect();
        path.extend(parents.iter().copied());
        let table = descend(&mut root, &path, line_no)?;
        table.insert(last.to_string(), value);
    }
    Ok(root)
}

/// Parse a TOML file.
pub fn parse_toml_file(path: &str) -> Result<BTreeMap<String, TomlValue>, TomlError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| TomlError { line: 0, msg: format!("{path}: {e}") })?;
    parse_toml(&text)
}
