//
// Test profile selection (the Rust port of julia/tutor/common/profile.jl).
//

/// The active test profile ("quick" or "release") from KB_TEST_PROFILE.
pub fn kb_test_profile() -> Result<&'static str, String> {
    let p = std::env::var("KB_TEST_PROFILE").unwrap_or_else(|_| "quick".to_string());
    if p != "quick" && p != "release" {
        return Err(format!(
            "KB_TEST_PROFILE must be `quick` or `release`, got `{p}`"
        ));
    }
    Ok(if p == "release" { "release" } else { "quick" })
}

/// Whether the release-only parts should run.
pub fn kb_release() -> bool {
    matches!(kb_test_profile(), Ok("release"))
}
