# Each test is a plain gtest binary with its own main (MPIHelper + TimedLogger).
set(_ldg_ns_tests fstheta_coefficients kovasznay taylor_green)
foreach(_test IN LISTS _ldg_ns_tests)
  add_executable(test_${_test} ${_test}.cc)
  target_link_libraries(test_${_test} PRIVATE ${DUNE_LIBS} GTest::gtest)
  add_test(NAME ${_test} COMMAND test_${_test})
  set_tests_properties(${_test} PROPERTIES LABELS "dune-ldg-ns-test")
endforeach()
# the temporal convergence study takes a few minutes in Release builds
set_tests_properties(taylor_green kovasznay PROPERTIES TIMEOUT 1800)
