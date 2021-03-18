This is a simple benchmark comparing performance of `Common::Atof`
and `Common::AtofPrecise` when used in `CSVParser`.

Just run `./run_parser_benchmark.sh` in this directory.

The test script generates 20000 rows, 2000 columns csv, 840MB file size.

For this test, `Common::Atof` is much faster than `Common::AtofPrecise`.

Benchmark run output on Intel Xeon 2640 v3:

```
========== Benchmark run Atof parser ==========                                                                                                                                                                                                                                                                                                                                                                 real    0m2.027s                                                                                                                                                                                        user    0m1.822s
real    0m2.027s
user    0m1.822s
sys     0m0.204s

real    0m2.186s
user    0m1.998s
sys     0m0.188s

real    0m2.202s
user    0m2.010s
sys     0m0.192s

========== Benchmark run AtofPrecise parser ==========
real    0m6.556s
user    0m6.324s
sys     0m0.232s

real    0m6.648s
user    0m6.496s
sys     0m0.152s

real    0m6.912s
user    0m6.748s
sys     0m0.164s
```