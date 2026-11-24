// Project : KadanoffBaym
// File    : tests/io/kbref_schema/legacy.rs
// Purpose : legacy data acceptance and the C-style writer format
//

use super::*;

#[test]
fn legacy_data_files_are_still_accepted() {
    let data = format!("{}/data/test", env!("CARGO_MANIFEST_DIR"));
    let mut files: Vec<String> = Vec::new();
    for entry in fs::read_dir(&data).unwrap() {
        let f = entry.unwrap().file_name().into_string().unwrap();
        if f.ends_with(".dat")
            && !f.starts_with("p2_")
            && !f.starts_with("phase7_")
            && !f.starts_with("p8_")
            && !f.starts_with("hubbard_")
            && !f.starts_with("fkm_")
            && !f.starts_with("holstein_")
        {
            files.push(f);
        }
    }
    assert_eq!(files.len(), 21);
    for f in &files {
        let d = read_kbref(&format!("{data}/{f}")).unwrap_or_else(|e| panic!("{f}: {e}"));
        assert!(
            d.metadata["data_kind"] == "weights" || d.metadata["data_kind"] == "herm_matrix",
            "{}: unexpected kind {}",
            f,
            d.metadata["data_kind"]
        );
    }
}

#[test]
fn fmt_writer_uses_c_style_floats() {
    // write_kbref must produce C-compatible %.17e output (checked
    // against the fmt unit test; here we only assert the line shape)
    let dir = tempdir_path("fmt");
    let p = dir.join("fmt.dat");
    let rows = vec![row("greg", 1, 0, 0, -1, 0.5, 0.0)];
    write_kbref(
        p.to_str().unwrap(),
        "weights",
        &[("case".to_string(), "x".to_string())],
        &rows,
    )
    .unwrap();
    let text = fs::read_to_string(&p).unwrap();
    let last = text.lines().last().unwrap();
    assert_eq!(
        last,
        format!("greg 1 0 0 -1 {} {}", fmt_e17(0.5), fmt_e17(0.0))
    );
    let _ = fs::remove_dir_all(&dir);
}
