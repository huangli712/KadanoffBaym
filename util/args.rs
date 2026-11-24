// Project : KadanoffBaym
// File    : util/args.rs
// Purpose : the command line helpers of the benchmark drivers

/// Integer value of the option `name` on the command line; `default`
/// when the option is absent or its value does not parse.
pub fn arg_or(name: &str, default: i64) -> i64 {
    let args: Vec<String> = std::env::args().collect();
    let mut val = default;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if a == name
            && let Some(v) = it.next()
        {
            val = v.parse().unwrap_or(default);
        }
    }
    val
}
