
### Cartpole trains

* Took about 1hr and 20k episodes only using an MLP

```
self._net = snt.DeepRNN(
            [
                snt.Flatten(),
                snt.LSTM(50),
                snt.nets.MLP([50, 50, action_spec.num_values]),
                #snt.VanillaRNN(16),
                #snt.VanillaRNN(16),
                snt.nets.MLP([64, 64, action_spec.num_values]),
            ]
        )
```

```
$ python cartpole.py
2021-06-14 17:16:47.099446: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2021-06-14 17:16:47.099557: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
[reverb/cc/platform/tfrecord_checkpointer.cc:150]  Initializing TFRecordCheckpointer in /tmp/tmpqai10lzb.
[reverb/cc/platform/tfrecord_checkpointer.cc:378] Loading latest checkpoint from /tmp/tmpqai10lzb
[reverb/cc/platform/default/server.cc:54] Started replay server on port 15091
2021-06-14 17:16:53.553107: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2021-06-14 17:16:53.553294: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
2021-06-14 17:16:53.554812: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (3228fafda2f1): /proc/driver/nvidia/version does not exist
2021-06-14 17:16:56.302609: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2021-06-14 17:16:56.306345: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2292540000 Hz
2021-06-14 17:16:56.338261: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
15.50 after 10 episodes
13.00 after 120 episodes
[reverb/cc/client.cc:162] Sampler and server are owned by the same process (245) so Table priority_table is accessed directly without gRPC.
[reverb/cc/client.cc:162] Sampler and server are owned by the same process (245) so Table priority_table is accessed directly without gRPC.
[reverb/cc/client.cc:162] Sampler and server are owned by the same process (245) so Table priority_table is accessed directly without gRPC.
[reverb/cc/client.cc:162] Sampler and server are owned by the same process (245) so Table priority_table is accessed directly without gRPC.
9.40 after 230 episodes
9.40 after 340 episodes
9.60 after 450 episodes
9.20 after 560 episodes
9.70 after 670 episodes
9.20 after 780 episodes
9.60 after 890 episodes
9.10 after 1000 episodes
9.20 after 1110 episodes
9.60 after 1220 episodes
9.30 after 1330 episodes
9.20 after 1440 episodes
9.80 after 1550 episodes
8.80 after 1660 episodes
9.10 after 1770 episodes
9.50 after 1880 episodes
9.90 after 1990 episodes
10.00 after 2100 episodes
9.20 after 2210 episodes
9.10 after 2320 episodes
9.30 after 2430 episodes
9.30 after 2540 episodes
9.90 after 2650 episodes
9.40 after 2760 episodes
9.40 after 2870 episodes
9.60 after 2980 episodes
9.70 after 3090 episodes
9.30 after 3200 episodes
9.70 after 3310 episodes
9.60 after 3420 episodes
9.00 after 3530 episodes
9.30 after 3640 episodes
9.30 after 3750 episodes
9.50 after 3860 episodes
9.30 after 3970 episodes
9.40 after 4080 episodes
9.30 after 4190 episodes
9.20 after 4300 episodes
9.10 after 4410 episodes
9.10 after 4520 episodes
9.60 after 4630 episodes
9.40 after 4740 episodes
9.40 after 4850 episodes
9.40 after 4960 episodes
9.30 after 5070 episodes
9.20 after 5180 episodes
9.50 after 5290 episodes
9.40 after 5400 episodes
8.90 after 5510 episodes
9.60 after 5620 episodes
9.80 after 5730 episodes
9.70 after 5840 episodes
9.40 after 5950 episodes
9.30 after 6060 episodes
9.50 after 6170 episodes
9.30 after 6280 episodes
9.00 after 6390 episodes
9.50 after 6500 episodes
9.40 after 6610 episodes
8.90 after 6720 episodes
9.60 after 6830 episodes
9.80 after 6940 episodes
9.30 after 7050 episodes
9.30 after 7160 episodes
9.20 after 7270 episodes
9.40 after 7380 episodes
9.50 after 7490 episodes
9.10 after 7600 episodes
9.60 after 7710 episodes
9.90 after 7820 episodes
9.30 after 7930 episodes
9.30 after 8040 episodes
9.80 after 8150 episodes
9.40 after 8260 episodes
9.70 after 8370 episodes
9.40 after 8480 episodes
9.60 after 8590 episodes
9.40 after 8700 episodes
9.60 after 8810 episodes
9.70 after 8920 episodes
10.00 after 9030 episodes
9.80 after 9140 episodes
9.30 after 9250 episodes
9.30 after 9360 episodes
9.80 after 9470 episodes
9.60 after 9580 episodes
9.30 after 9690 episodes
9.70 after 9800 episodes
9.00 after 9910 episodes
9.50 after 10020 episodes
9.40 after 10130 episodes
9.30 after 10240 episodes
9.40 after 10350 episodes
9.30 after 10460 episodes
9.20 after 10570 episodes
9.40 after 10680 episodes
9.40 after 10790 episodes
9.50 after 10900 episodes
9.60 after 11010 episodes
9.30 after 11120 episodes
9.70 after 11230 episodes
9.20 after 11340 episodes
9.60 after 11450 episodes
9.70 after 11560 episodes
11.00 after 11670 episodes
12.40 after 11780 episodes
13.40 after 11890 episodes
14.50 after 12000 episodes
15.30 after 12110 episodes
16.30 after 12220 episodes
16.70 after 12330 episodes
16.50 after 12440 episodes
16.90 after 12550 episodes
15.60 after 12660 episodes
15.40 after 12770 episodes
16.30 after 12880 episodes
16.30 after 12990 episodes
15.30 after 13100 episodes
16.00 after 13210 episodes
15.90 after 13320 episodes
15.50 after 13430 episodes
15.70 after 13540 episodes
16.40 after 13650 episodes
15.20 after 13760 episodes
13.00 after 13870 episodes
15.40 after 13980 episodes
18.60 after 14090 episodes
19.10 after 14200 episodes
19.20 after 14310 episodes
20.30 after 14420 episodes
18.30 after 14530 episodes
19.30 after 14640 episodes
18.60 after 14750 episodes
18.10 after 14860 episodes
19.50 after 14970 episodes
20.30 after 15080 episodes
20.20 after 15190 episodes
19.10 after 15300 episodes
18.50 after 15410 episodes
18.60 after 15520 episodes
18.90 after 15630 episodes
18.80 after 15740 episodes
20.20 after 15850 episodes
19.60 after 15960 episodes
20.20 after 16070 episodes
19.70 after 16180 episodes
22.10 after 16290 episodes
21.30 after 16400 episodes
21.40 after 16510 episodes
22.10 after 16620 episodes
22.10 after 16730 episodes
22.00 after 16840 episodes
24.30 after 16950 episodes
25.40 after 17060 episodes
25.50 after 17170 episodes
28.40 after 17280 episodes
25.50 after 17390 episodes
31.50 after 17500 episodes
28.00 after 17610 episodes
30.30 after 17720 episodes
31.70 after 17830 episodes
31.00 after 17940 episodes
36.90 after 18050 episodes
31.30 after 18160 episodes
28.90 after 18270 episodes
84.80 after 18380 episodes
103.40 after 18490 episodes
64.50 after 18600 episodes
132.80 after 18710 episodes
137.60 after 18820 episodes
124.90 after 18930 episodes
137.80 after 19040 episodes
187.20 after 19150 episodes
191.80 after 19260 episodes
207.60 after 19370 episodes
219.30 after 19480 episodes
241.50 after 19590 episodes
262.80 after 19700 episodes
441.40 after 19810 episodes
500.00 after 19920 episodes
500.00 after 20030 episodes
500.00 after 20140 episodes
30.80 after 20250 episodes
333.10 after 20360 episodes
392.00 after 20470 episodes
411.60 after 20580 episodes
471.90 after 20690 episodes
```
