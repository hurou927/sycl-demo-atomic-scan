#include <CL/sycl.hpp>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/atomic.hpp>
#include <CL/sycl/group_algorithm.hpp>
#include <CL/sycl/memory_enums.hpp>
#include <CL/sycl/nd_item.hpp>
#include <algorithm>
#include <ext/oneapi/atomic_ref.hpp>
#include <ext/oneapi/group_algorithm.hpp>
#include <ext/oneapi/reduction.hpp>
#include <helpers/timestamp.hpp>
#include <iostream>
using namespace cl::sycl;
using namespace std;

constexpr unsigned int log2(unsigned int n) {
  return n <= 1 ? 0 : 1 + log2((n + 1) / 2);
}
constexpr unsigned int NUM_THREADS_PER_GROUP = 512;
constexpr unsigned int LOG2_NUM_THREADS_PER_GROUP = log2(NUM_THREADS_PER_GROUP);

template <typename T> void hostInlineScan(T *host_input_buf, size_t num_items) {
  for (int i = 1; i < num_items; i++) {
    host_input_buf[i] += host_input_buf[i - 1];
  }
}

template <typename T>
void deviceInlineScan(queue &q, T *host_input_buf, size_t num_items) {

  auto num_groups =
      (num_items + NUM_THREADS_PER_GROUP - 1) / NUM_THREADS_PER_GROUP;
  auto num_threads = num_groups * NUM_THREADS_PER_GROUP;

  T *DEVICE_RESULT = static_cast<T *>(malloc_device(num_items * sizeof(T), q));
  T *DEVICE_GROUP_SUM =
      static_cast<T *>(malloc_device(num_groups * sizeof(T), q));
  /* int *DEVICE_GROUP_FLAG = */
  /*     static_cast<int *>(malloc_device(num_groups * sizeof(int), q)); */
  int *DEVICE_DYNAMIC_GROUP_ID =
      static_cast<int *>(malloc_device(1 * sizeof(int), q));
  int *DEVICE_COUNTER = static_cast<int *>(malloc_device(1 * sizeof(int), q));
  q.memset(DEVICE_DYNAMIC_GROUP_ID, 0, sizeof(int) * 1);
  q.memset(DEVICE_COUNTER, 0, sizeof(int) * 1);
  /* q.memset(DEVICE_GROUP_FLAG, 0, sizeof(int) * num_groups); */

  q.memcpy(DEVICE_RESULT, host_input_buf, sizeof(T) * num_items).wait();
  event kernel_event = q.submit([&](handler &cgh) {
    // https://support.codeplay.com/t/compile-error-reference-to-non-static-member-function-must-be-called/383
    /* auto deviceBuf = buf.template get_access<access::mode::read_write>(cgh);
     */

    auto localRange = range<1>(NUM_THREADS_PER_GROUP);
    accessor<T, 1, access::mode::read_write, access::target::local>
        LOCAL_SCAN_SPACE(localRange, cgh);
    auto oneRange = range<1>(1);
    accessor<T, 1, access::mode::read_write, access::target::local> LOCAL_SUM(
        oneRange, cgh);
    accessor<int, 1, access::mode::read_write, access::target::local>
        LOCAL_GROUP_ID(oneRange, cgh);

    auto kernel = [=](nd_item<1> it) {
      const auto local_id = it.get_local_id(0);

      int is_worker_thread = local_id == 0 ? 1 : 0;
      // ===========================================
      // Calc group id dynamicaly using atomic add
      // * v_group_id: dynamic group id
      // * v_global_id: dunamic global id
      // ===========================================
      int v_group_id;
      if (is_worker_thread) {
        ext::oneapi::atomic_ref<int, ext::oneapi::memory_order::relaxed,
                                ext::oneapi::memory_scope::device,
                                access::address_space::global_space>
            ATOMIC_GROUP_ID(DEVICE_DYNAMIC_GROUP_ID[0]);
        v_group_id = ATOMIC_GROUP_ID.fetch_add(1);
        LOCAL_GROUP_ID[0] = v_group_id;
      }
      it.barrier(access::fence_space::local_space);
      v_group_id = LOCAL_GROUP_ID[0];
      int v_global_id = v_group_id * NUM_THREADS_PER_GROUP + local_id;

      //=============================================
      // Load item from device memory
      //=============================================
      T local_item;
      if (v_global_id < num_items) {
        local_item = DEVICE_RESULT[v_global_id];
      } else {
        local_item = 0;
      }

      // ============================================
      // Local scan
      // ============================================
      /* T local_inclusive_prefix_sum = local_item; */

      /* LOCAL_SCAN_SPACE[local_id] = local_inclusive_prefix_sum; */

      /* for (auto offset = 1; offset < NUM_THREADS_PER_GROUP; */
      /*      offset = offset << 1) { */
      /*   int needUpdate = local_id >= offset ? 1 : 0; */
      /*   it.barrier(access::fence_space::local_space); */
      /*   if (needUpdate != 0) { */
      /*     local_inclusive_prefix_sum += LOCAL_SCAN_SPACE[local_id - offset]; */
      /*   } */
      /*   it.barrier(access::fence_space::local_space); */
      /*   if (needUpdate != 0) { */
      /*     LOCAL_SCAN_SPACE[local_id] = local_inclusive_prefix_sum; */
      /*   } */
      /* } */
      /* it.barrier(access::fence_space::local_space); */

      T local_inclusive_prefix_sum = inclusive_scan_over_group(it.get_group(), local_item, cl::sycl::plus<>());
      if(local_id == NUM_THREADS_PER_GROUP - 1) {
        LOCAL_SCAN_SPACE[local_id] = local_inclusive_prefix_sum;
      }
      it.barrier(access::fence_space::local_space);

      //================================================
      // Scan of group's sum
      //================================================

      if (is_worker_thread) {

        // Ref:
        //  DATA PARALLEL C++: MASTERING DPC++ FOR PROGRAMMING OF HETEROGENEOUT
        //  SYSTEMS YSING C++ AND SYCL P.517 Implementing Device-Wide
        //  Synchronization
        ext::oneapi::atomic_ref<int, ext::oneapi::memory_order::acq_rel,
                                ext::oneapi::memory_scope::device,
                                access::address_space::global_space>
            ATOMIC_COUNTER(DEVICE_COUNTER[0]);
        /* ext::oneapi::atomic_ref<int, ext::oneapi::memory_order::acq_rel, */
        /*                         ext::oneapi::memory_scope::device, */
        /*                         access::address_space::global_space> */
        /*     ATOMIC_STORE(DEVICE_GROUP_FLAG[v_group_id]); */
        T group_sum = LOCAL_SCAN_SPACE[NUM_THREADS_PER_GROUP - 1];
        if (v_group_id == 0) {
          DEVICE_GROUP_SUM[v_group_id] = group_sum;
          ATOMIC_COUNTER++;
          /* ATOMIC_STORE.store(1); */
          LOCAL_SUM[0] = 0;
        } else {
          /* ext::oneapi::atomic_ref<int, ext::oneapi::memory_order::acq_rel, */
          /*                         ext::oneapi::memory_scope::device, */
          /*                         access::address_space::global_space> */
          /*     ATOMIC_LOAD(DEVICE_GROUP_FLAG[v_group_id - 1]); */
          /* while (ATOMIC_LOAD.load() != 1) { */
          /* } */
          while (ATOMIC_COUNTER.load() != v_group_id) {
          }

          auto exclusive_group_prefix_sum = DEVICE_GROUP_SUM[v_group_id - 1];
          DEVICE_GROUP_SUM[v_group_id] = exclusive_group_prefix_sum + group_sum;
          /* ATOMIC_STORE.store(1); */
          ATOMIC_COUNTER++;
          LOCAL_SUM[0] = exclusive_group_prefix_sum;
        }
      }

      it.barrier(access::fence_space::local_space);

      //==================================================
      // Global scan and write result
      //==================================================
      if (v_global_id < num_items) {
        DEVICE_RESULT[v_global_id] =
            local_inclusive_prefix_sum - local_item + LOCAL_SUM[0];
      }
    };
    cgh.parallel_for<class pm>(nd_range<1>{range<1>(num_items), localRange},
                               kernel);
  });
  q.wait();
  q.memcpy(host_input_buf, DEVICE_RESULT, sizeof(T) * num_items);

  /* int *h = static_cast<T *>(malloc(sizeof(int) * num_groups)); */
  /* q.memcpy(h, DEVICE_GROUP_FLAG, sizeof(T) * num_groups).wait(); */
  /* for (int i = 0; i < num_groups; i++) */
  /*   cout << h[i] << ","; */
  /* cout << "\n"; */
  /* free(h); */
  free(DEVICE_RESULT, q);
  free(DEVICE_GROUP_SUM, q);
  free(DEVICE_DYNAMIC_GROUP_ID, q);

  auto end =
      kernel_event.get_profiling_info<info::event_profiling::command_end>();
  auto start =
      kernel_event.get_profiling_info<info::event_profiling::command_start>();
  
  std::cout << "kernel execution time without submission: "
            <<  ((end - start) / 1.0e6) << " ms\n";
  std::cout << "kernel execution time without submission: "
            << static_cast<double>(sizeof(T)) * num_items / 1024 / 1024 / 1024 / ((end - start) / 1.0e9) << " GB/s\n";
}

int main(int argc, char *argv[]) {

  auto t = TimeStamp<std::string>();
  size_t num_items = 1024 * 16;
  if (argc == 2) {
    num_items = atoi(argv[1]) * 1024 * 1024;
  }

  int *data = static_cast<int *>(malloc(num_items * sizeof(int)));
  for (int i = 0; i < num_items; i++)
    data[i] = 1;

  gpu_selector d_selector;
  queue q(d_selector);
  cout << "Running on device: " << q.get_device().get_info<info::device::name>()
       << "\n";
  t.stamp("device");
  deviceInlineScan<int>(q, data, num_items);
  t.stamp("host");
  hostInlineScan<int>(data, num_items);
  t.stamp();

  cout << "num_items: " << num_items << "\n";
  t.print();
  for (int i = 0; i < 10; i++)
    cout << data[i] << ",";
  cout << "...,";
  for (int i = num_items - 10; i < num_items; i++)
    cout << data[i] << ",";
  cout << "\n";

  free(data);
  return 0;
}
