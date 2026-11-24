// Project : KadanoffBaym
// File    : tests/io/ref_schema/legacy.rs
// Purpose : legacy data acceptance and the printf-style writer format
//

use std::fs::{read_dir, read_to_string, remove_dir_all};

use super::*;

#[test]
fn legacy_data_files_are_still_accepted() {
    let data = format!("{}/data/test", env!("CARGO_MANIFEST_DIR"));
    let mut files: Vec<String> = Vec::new();
    for entry in read_dir(&data).unwrap() {
        let f = entry.unwrap().file_name().into_string().unwrap();
        if f.ends_with(".dat")
            && !f.starts_with("convolution_utilities_")
            && !f.starts_with("differentiation_")
            && !f.starts_with("response_")
            && !f.starts_with("utilities_")
            && !f.starts_with("moving_")
            && f != "fourier.dat"
            && f != "steady.dat"
            && !f.starts_with("hubbard_")
            && !f.starts_with("fkm_")
            && !f.starts_with("holstein_")
        {
            files.push(f);
        }
    }
    assert_eq!(files.len(), 21);
    for f in &files {
        let d = read_ref(&format!("{data}/{f}")).unwrap_or_else(|e| panic!("{f}: {e}"));
        assert!(
            d.metadata["data_kind"] == "weights" || d.metadata["data_kind"] == "herm_matrix",
            "{}: unexpected kind {}",
            f,
            d.metadata["data_kind"]
        );
    }
}

#[test]
fn fmt_writer_uses_printf_floats() {
    // write_ref must produce printf-compatible %.17e output (checked
    // against the fmt unit test; here we only assert the line shape)
    let dir = tempdir_path("fmt");
    let p = dir.join("fmt.dat");
    let rows = vec![reference_row("greg", 1, 0, 0, -1, 0.5, 0.0)];
    write_ref(
        p.to_str().unwrap(),
        "weights",
        &[("case".to_string(), "x".to_string())],
        &rows,
    )
    .unwrap();
    let text = read_to_string(&p).unwrap();
    let last = text.lines().last().unwrap();
    assert_eq!(
        last,
        format!("greg 1 0 0 -1 {} {}", fmt_e17(0.5), fmt_e17(0.0))
    );
    let _ = remove_dir_all(&dir);
}
