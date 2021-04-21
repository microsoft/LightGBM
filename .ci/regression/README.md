# Regression Test

## typical structure

```
.
├── sample
│   ├── ref_data
│   │   ├── 1.ref
│   │   └── 2.ref
│   ├── sample_1.sh
│   └── sample_test2.sh (generates `data/{1.ref, 2.ref}` to compare with in `ref_data`)
└── verify_result.sh
```

## steps

### create <task_dir> and add test job to ci
in example `sample`


### create referance truth data
in `<task>/ref_data`
in example `1.ref` and `2.ref`
could be bin file model or any file


### create test script
Should generate `1.ref` and `2.ref` in `$1` dir
in example `sample_1` and `sample_test2.sh


### test run
Run `bash ../verify_result.sh` in <task_dir> as working directory
