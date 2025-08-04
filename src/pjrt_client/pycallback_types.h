#ifndef XLA_PJRT_MULTIMESH_PY_CALLBACK_TYPES_H_
#define XLA_PJRT_MULTIMESH_PY_CALLBACK_TYPES_H_

#include <vector>
#include <optional>
#include <string>
#include <tuple>
#include <functional>

namespace multimesh {

struct TaskOptions {
  std::optional<std::string> name{};
  std::vector<int64_t> devices{};
  std::vector<int64_t> dims{};
  std::vector<std::string> axes{};
  std::vector<std::pair<std::string, std::string>> logical_axes{};
  std::optional<std::pair<std::string, std::string>> split_backprop{};
  std::function<std::vector<int64_t>(int64_t iteration)>
      loop_dependent_devices{};
};

using TaskOptionsTuple =
    std::tuple<std::string, std::vector<int64_t>, std::vector<int64_t>,
               std::vector<std::string>,
               std::vector<std::pair<std::string, std::string>>,
               std::optional<std::pair<std::string, std::string>>,
               std::function<std::vector<int64_t>(int64_t iteration)>>;

}  // namespace multimesh

extern "C" void RegisterMetadataNameTask(
    std::string matcher,
    std::function<multimesh::TaskOptions(const std::string&, bool)> callback);

#endif