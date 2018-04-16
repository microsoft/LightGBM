Parallel Learning Example
=========================

Here is an example for LightGBM to perform parallel learning for 2 machines.

1. Edit mlist.txt, write the ip of these 2 machines that you want to run application on.

  ```
  machine1_ip 12400
  machine2_ip 12400
  ```

2. Copy this folder and executable file to these 2 machines that you want to run application on.

3. Run command in this folder on both 2 machines:

   For Windows: ```lightgbm.exe config=train.conf```

   For Linux: ```./lightgbm config=train.conf```

This parallel learning example is based on socket. LightGBM also support parallel learning based on mpi.

For more details about the usage of parallel learning, please refer to [this](https://github.com/Microsoft/LightGBM/blob/master/docs/Parallel-Learning-Guide.rst).
