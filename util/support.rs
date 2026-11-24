// Project : KadanoffBaym
// File    : util/support.rs
// Purpose : the small helpers shared by the integration tests (complex
//           literals, panic checks, fresh temporary directories and
//           file writing)
//

use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use kadanoffbaym::complex::Complex;

/// `Complex::new(re, im)` under the short name of the test blueprints.
pub fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

/// Whether `f` panics.
pub fn panics<F: FnOnce() -> R, R>(f: F) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
}

/// A fresh temporary directory tagged with `tag`.
pub fn tmpdir(tag: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "kb-{tag}-{}-{nanos}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

/// Write `text` to `<dir>/<name>` and return the path.
pub fn write_file(dir: &str, name: &str, text: &str) -> String {
    let path = format!("{dir}/{name}");
    fs::write(&path, text).unwrap();
    path
}
