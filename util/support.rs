// Project : KadanoffBaym
// File    : util/support.rs
// Purpose : the small helpers shared by the integration tests (complex
//           literals, panic checks, fresh temporary directories and
//           file writing)
//

use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use std::time::UNIX_EPOCH;

use std::env::{current_exe, temp_dir};
use std::fs::{create_dir_all, write};
use std::panic::catch_unwind;
use std::process::id;

use kadanoffbaym::complex::Complex;

/// `Complex::new(re, im)` under the short name of the test references.
pub fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

/// Whether `f` panics.
pub fn panics<F: FnOnce() -> R, R>(f: F) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_err()
}

/// The workspace root: the nearest ancestor with `Cargo.toml` and `data/tutor`.
pub fn workspace_root() -> PathBuf {
    let exe = current_exe().expect("current exe");
    let mut dir = exe.parent().expect("exe dir").to_path_buf();
    loop {
        if dir.join("Cargo.toml").is_file() && dir.join("data").join("tutor").is_dir() {
            return dir;
        }
        dir = match dir.parent() {
            Some(p) => p.to_path_buf(),
            None => panic!("workspace root with data/tutor not found"),
        };
    }
}

/// A fresh temporary directory tagged with `tag`.
pub fn tmpdir(tag: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = temp_dir().join(format!(
        "kb-{tag}-{}-{nanos}-{}",
        id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

/// Write `text` to `<dir>/<name>` and return the path.
pub fn write_file(dir: &str, name: &str, text: &str) -> String {
    let path = format!("{dir}/{name}");
    write(&path, text).unwrap();
    path
}
