//
// Project : KadanoffBaym
// File    : src/basics/blocks/copyzone.rs
// Purpose : rectangular orbital region driving the block copies
//
// Blueprint: julia/src/blocks/copyzone.jl.

/// Rectangular orbital region `[x1..=x2, y1..=y2]` for copying
/// (blueprint `CopyZone`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyZone {
    pub x1: i64,
    pub y1: i64,
    pub x2: i64,
    pub y2: i64,
}

impl CopyZone {
    /// Single-cell zone.
    pub fn point(x: i64, y: i64) -> CopyZone {
        CopyZone {
            x1: x,
            y1: y,
            x2: x,
            y2: y,
        }
    }

    /// Square zone `[x..x+delta-1, y..y+delta-1]` (`delta >= 1`).
    pub fn square(x: i64, y: i64, delta: i64) -> CopyZone {
        assert!(delta >= 1, "CopyZone::square: delta >= 1, got {delta}");
        CopyZone {
            x1: x,
            y1: y,
            x2: x + delta - 1,
            y2: y + delta - 1,
        }
    }

    /// Validity: `x2 >= x1 >= 0` and `y2 >= y1 >= 0`.
    pub fn is_valid(&self) -> bool {
        self.x2 >= self.x1 && self.x1 >= 0 && self.y2 >= self.y1 && self.y1 >= 0
    }

    /// Number of orbital rows in the zone.
    pub(crate) fn nrows(&self) -> i64 {
        self.x2 - self.x1 + 1
    }

    /// Number of orbital columns in the zone.
    pub(crate) fn ncols(&self) -> i64 {
        self.y2 - self.y1 + 1
    }

    /// Two zones are compatible when their shapes match.
    pub fn compatible(a: CopyZone, b: CopyZone) -> bool {
        a.nrows() == b.nrows() && a.ncols() == b.ncols()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zone_validity_and_shapes() {
        assert!(CopyZone::point(0, 0).is_valid());
        assert!(CopyZone::square(1, 2, 3).is_valid());
        assert_eq!(CopyZone::square(1, 2, 3).x2, 3);
        assert!(!CopyZone::point(-1, 0).is_valid());
        assert!(CopyZone::compatible(
            CopyZone::square(0, 0, 2),
            CopyZone::square(1, 1, 2)
        ));
        assert!(!CopyZone::compatible(
            CopyZone::square(0, 0, 2),
            CopyZone::square(1, 1, 3)
        ));
    }
}
