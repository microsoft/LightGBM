#!cmake -P
file(WRITE split_test_helper.h "")
foreach(i RANGE 1 999)
  file(APPEND split_test_helper.h
    "#if defined(EIGEN_TEST_PART_${i}) || defined(EIGEN_TEST_PART_ALL)\n"
    "#define CALL_SUBTEST_${i}(FUNC) CALL_SUBTEST(FUNC)\n"
    "#else\n"
    "#define CALL_SUBTEST_${i}(FUNC)\n"
    "#endif\n\n"
  )
endforeach()