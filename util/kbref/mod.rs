// Project : KadanoffBaym
// File    : util/kbref/mod.rs
// Purpose : the kbref-v1 plain-text reference format: the reader (with
//           strict schema validation) and the writer (schema
//           round-trip), split by concern
//

// Index translation (NESSi is 0-based): mat[q] -> G.mat[q], ret[i,j] ->
// G.ret[i,j] with 0 <= j <= i <= nt, tv[i,q] -> G.lmix[i,q], les[i,j] ->
// G.less[i,j] with 0 <= i <= j <= nt.
//
// Layout: `error` (the parse/validation error), `schema` (the kind,
// metadata, and component tables), `check` (the index-domain validation
// of one row), `read` (the row and file types and the reader), `write`
// (the writer).
//

mod check;
mod error;
mod read;
mod schema;
mod write;

pub use read::{KbrefData, KbrefRow, read_kbref};
pub use schema::{
    KBREF_CG_SCALARS, KBREF_OBSERVABLE_META, KBREF_SELECTED_GREEN_KEYS, KBREF_SERIES_CASE_SETS,
    OBSERVABLE_MODELS, approximations, observable_class, observable_components, series_class,
    series_components, series_generator,
};
pub use write::write_kbref;
