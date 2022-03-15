#include <algorithm>
#include <CL/sycl.hpp>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/atomic.hpp>
#include <CL/sycl/memory_enums.hpp>
#include <CL/sycl/nd_item.hpp>
#include <ext/oneapi/atomic_ref.hpp>
#include <ext/oneapi/reduction.hpp>
#include <iostream>
using namespace cl::sycl;
using namespace std;

constexpr unsigned int log2(unsigned int n) {
  return n <= 1 ? 0 : 1 + log2((n + 1) / 2);
}
constexpr unsigned int numThreadsPerGroup = 64;
constexpr unsigned int log2NumThreadsPerGroup = log2(numThreadsPerGroup);

template <typename T>
void deviceInlineScan(queue &q, T *hostBuf, size_t numItems) {

  auto numGroups = (numItems + numThreadsPerGroup - 1) / numThreadsPerGroup;
  auto numThreads = numGroups * numThreadsPerGroup;

  T *DEVICE_RESULT = static_cast<T *>(malloc_device(numItems * sizeof(T), q));
  T *DEVICE_GROUP_SUM =
      static_cast<T *>(malloc_device(numGroups * sizeof(T), q));
  int *DEVICE_DYNAMIC_GROUP_ID =
      static_cast<int *>(malloc_device(1 * sizeof(int), q));
  int *DEVICE_COUNTER = static_cast<int *>(malloc_device(1 * sizeof(int), q));
  q.memset(DEVICE_DYNAMIC_GROUP_ID, 0, sizeof(int) * 1);
  q.memset(DEVICE_COUNTER, 0, sizeof(int) * 1);

  q.memcpy(DEVICE_RESULT, hostBuf, sizeof(T) * numItems).wait();
  q.submit([&](handler &cgh) {
    // https://support.codeplay.com/t/compile-error-reference-to-non-static-member-function-must-be-called/383
    /* auto deviceBuf = buf.template get_access<access::mode::read_write>(cgh);
     */

    auto localRange = range<1>(numThreadsPerGroup);
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
      int v_global_id = v_group_id * numThreadsPerGroup + local_id;

      //=============================================
      // Load item from device memory
      //=============================================
      T local_item;
      if (v_global_id < numItems) {
        local_item = DEVICE_RESULT[v_global_id];
      } else {
        local_item = 0;
      }

      // ============================================
      // Local scan
      // ============================================
      T local_inclusive_prefix_sum = local_item;

      LOCAL_SCAN_SPACE[local_id] = local_inclusive_prefix_sum;

      for (auto offset = 1; offset < numThreadsPerGroup; offset = offset << 1) {
        int needUpdate = local_id >= offset ? 1 : 0;
        it.barrier(access::fence_space::local_space);
        if (needUpdate != 0) {
          local_inclusive_prefix_sum += LOCAL_SCAN_SPACE[local_id - offset];
        }
        it.barrier(access::fence_space::local_space);
        if (needUpdate != 0) {
          LOCAL_SCAN_SPACE[local_id] = local_inclusive_prefix_sum;
        }
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
        T group_sum = LOCAL_SCAN_SPACE[numThreadsPerGroup - 1];
        if (v_group_id == 0) {
          DEVICE_GROUP_SUM[v_group_id] = group_sum;
          ATOMIC_COUNTER++;
          LOCAL_SUM[0] = 0;
        } else {
          while (ATOMIC_COUNTER.load() != v_group_id) {
          }
          auto exclusive_group_prefix_sum = DEVICE_GROUP_SUM[v_group_id - 1];
          DEVICE_GROUP_SUM[v_group_id] = exclusive_group_prefix_sum + group_sum;
          ATOMIC_COUNTER++;
          LOCAL_SUM[0] = exclusive_group_prefix_sum;
        }
      }

      it.barrier(access::fence_space::local_space);

      //==================================================
      // Global scan and write result
      //==================================================
      if (v_global_id < numItems) {
        DEVICE_RESULT[v_global_id] =
            local_inclusive_prefix_sum - local_item + LOCAL_SUM[0];
      }
    };
    cgh.parallel_for<class pm>(nd_range<1>{range<1>(numItems), localRange},
                               kernel);
  });
  q.wait();
  q.memcpy(hostBuf, DEVICE_RESULT, sizeof(T) * numItems);

  /* int *hGroupSum = static_cast<T *>(malloc(sizeof(int) * numGroups)); */
  /* q.memcpy(hGroupSum, dGroupSum, sizeof(T) * numGroups).wait(); */
  /* for (int i = 0; i < numGroups; i++) */
  /*   cout << hGroupSum[i] << ","; */
  /* cout << "\n"; */
  /* free(hGroupSum); */
  free(DEVICE_RESULT, q);
  free(DEVICE_DYNAMIC_GROUP_ID, q);
  free(DEVICE_DYNAMIC_GROUP_ID, q);
}

int main() {

  const size_t numItems = 1024*16;

  int *data = static_cast<int *>(malloc(numItems * sizeof(int)));
  for (int i = 0; i < numItems; i++)
    data[i] = 1;

  default_selector d_selector;
  queue q(d_selector);
  cout << "Running on device: " << q.get_device().get_info<info::device::name>()
       << "\n";

  deviceInlineScan<int>(q, data, numItems);


  for (int i = 0 ; i < 10; i++)
    cout << data[i] << ",";
  cout << "...,";
  for (int i = numItems - 10 ; i < numItems; i++)
    cout << data[i] << ",";
  cout << "\n";
  return 0;
}
