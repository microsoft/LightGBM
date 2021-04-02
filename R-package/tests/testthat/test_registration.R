# this test is used to catch silent errors in routine registration,
# like the one documented in
# https://github.com/microsoft/LightGBM/issues/4045#issuecomment-812289182
test_that("lightgbm routine registration worked", {
    dll_info <- getLoadedDLLs()[["lightgbm"]]

    # check that dynamic lookup has been disabled
    expect_false(dll_info[["dynamicLookup"]])

    # check that all the expected symbols show up
    registered_routines <- getDLLRegisteredRoutines(dll_info[["path"]])[[".Call"]]
    expect_gt(length(registered_routines), 20L)
})
