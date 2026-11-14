/// Test-profile selection (blueprint julia/test/common/profile.jl).

/// Active test profile ("quick" or "release").
pub fn test_profile() -> &'static str {
    match std::env::var("KB_TEST_PROFILE").ok().as_deref() {
        None | Some("quick") => "quick",
        Some("release") => "release",
        Some(other) => panic!("KB_TEST_PROFILE must be `quick` or `release`, got `{other}`"),
    }
}

/// Whether the release-only tests should run.
pub fn is_release() -> bool {
    test_profile() == "release"
}
