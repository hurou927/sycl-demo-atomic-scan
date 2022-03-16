#include <chrono>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using TimePointT = std::chrono::steady_clock::time_point;
using TimeUnit = std::chrono::nanoseconds;
using TimeDurationT = std::chrono::duration<double>;
constexpr auto timeUnitStr = "ms";

template <typename T> class Duration {
public:
  T name;
  TimeDurationT duration;

  Duration(T name, TimeDurationT duration) : name(name), duration(duration) {}
};

template <typename T> class TimeStamp {
private:
  std::optional<T> accumName = std::nullopt;
  TimePointT accumTime = std::chrono::steady_clock::now(); // immutable.
  const size_t capacity = 20;
  std::vector<Duration<T>> durations;

public:
  TimeStamp(){};

  void stamp(T name) {
    auto dp = std::chrono::steady_clock::now();
    if (accumName) {
      durations.push_back(Duration<T>(accumName.value(), dp - accumTime));
    }
    accumTime = dp;
    accumName = name;
  };
  void stamp() {
    auto dp = std::chrono::steady_clock::now();
    if (accumName) {
      durations.push_back(Duration<T>(accumName.value(), dp - accumTime));
    }
    accumTime = dp;
    accumName = std::nullopt;
  };

  std::vector<Duration<T>> getDurations() { return durations; };

  void print() {
    int c = 1;
    for (const auto &e : durations) {
      std::cout << c << ". " << e.name << ". "
                << std::chrono::duration_cast<TimeUnit>(e.duration).count() / 1e6
                << " " << timeUnitStr << std::endl;
      c++;
    }
  };
};

/* int main() { */
/*   auto t = TimeStamp<std::string>(); */

/*   t.stamp("1"); */
/*   std::this_thread::sleep_for(std::chrono::milliseconds(500)); */
/*   t.stamp(); */
/*   std::this_thread::sleep_for(std::chrono::milliseconds(1000)); */
/*   t.stamp("2"); */
/*   std::this_thread::sleep_for(std::chrono::milliseconds(500)); */
/*   t.stamp(); */

/*   t.print(); */
/*   return 0; */
/* } */
