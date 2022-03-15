# Sycl(DPC++/OneAPI) Demo: Scan

## Require

```sh
$ . /opt/intel/oneapi/setvars.sh
```

## Compile

```sh
$ make
```


## Algorithm

1. Each thread-group calculates its group-id and global-thread-id dynamically using the atomic counter.
2. Each thread loads data at global-thread-id as an index and calculates local scan.
3. A worker thread in each thread-group calculates the global prefix-sum of partial sums.
4. Each thread calculates the global scan by the result of step-3 and writes it to the device memory.

![atomic_scan](./image/atomic_scan.drawio.png)

