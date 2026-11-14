#
# profile.jl
#
# Test profile selection. The only accepted environment variable is
# KB_TEST_PROFILE = quick | release (default: quick).
#
#   quick   : analytic tests and small NESSi point-wise data; full
#             convergence grids and large memory gates are skipped.
#   release : all tests.
#

"""
    kb_test_profile()

Return the active test profile ("quick" or "release"). Any other value of
`KB_TEST_PROFILE` raises an `ArgumentError`.
"""
function kb_test_profile()
    p = get(ENV, "KB_TEST_PROFILE", "quick")
    p in ("quick", "release") ||
        throw(ArgumentError("KB_TEST_PROFILE must be `quick` or `release`, got `$p`"))
    return p
end

"""
    kb_release()

Whether the release-only tests should be run.
"""
kb_release() = kb_test_profile() == "release"
