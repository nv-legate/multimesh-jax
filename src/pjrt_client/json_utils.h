#ifndef XLA_PJRT_LEGATE_JSON_UTILS_H_
#define XLA_PJRT_LEGATE_JSON_UTILS_H_

#include <optional>
#include <string>
#include <vector>

#include "json/json.h"
#include "xla/util.h"

namespace xla {

bool GetTaskValue(const Json::Value& value, const char* name, bool& result);

bool GetTaskValue(const Json::Value& value, const char* name, int& result);

bool GetTaskValue(const Json::Value& value, const char* name, int64_t& result);

bool GetTaskValue(const Json::Value& value, const char* name,
                  std::string& result);

inline const auto& ToJson(const std::string& s) { return s; }

template <class T>
Json::Value ToJson(const std::vector<T>& vector) {
  Json::Value vec(Json::arrayValue);
  for (auto&& v : vector) {
    vec.append(ToJson(v));
  }
  return vec;
}

template <class T, size_t N>
Json::Value ToJson(const absl::InlinedVector<T, N>& vector) {
  Json::Value vec(Json::arrayValue);
  for (auto&& v : vector) {
    vec.append(ToJson(v));
  }
  return vec;
}

template <class T>
bool GetTaskValue(const Json::Value& value, const char* name,
                  std::vector<std::optional<T>>& result) {
  if (!value.isArray()) {
    return false;
  }
  result.reserve(value.size());
  for (const auto& sub_json : value) {
    if (sub_json.isNull()) {
      result.push_back(std::nullopt);
    } else {
      T t;
      if (!GetTaskValue(sub_json, name, t)) {
        return false;
      }
      result.push_back(std::move(t));
    }
  }
  return true;
}

template <class T, class U>
bool GetTaskValue(const Json::Value& value, const char* name,
                  std::pair<T, U>& result) {
  if (!value.isArray()) {
    return false;
  }

  if (value.size() != 2) {
    return false;
  }

  T t;
  if (!GetTaskValue(value[0], name, t)) {
    return false;
  }

  U u;
  if (!GetTaskValue(value[1], name, u)) {
    return false;
  }

  result = {t, u};
  return true;
}

template <class T>
bool GetTaskValue(const Json::Value& value, const char* name,
                  std::vector<T>& result) {
  if (!value.isArray()) {
    return false;
  }
  result.reserve(value.size());
  for (const auto& sub_json : value) {
    T t;
    if (!GetTaskValue(sub_json, name, t)) {
      return false;
    }
    result.push_back(std::move(t));
  }
  return true;
}

template <class T, size_t N>
bool GetTaskValue(const Json::Value& value, const char* name,
                  absl::InlinedVector<std::optional<T>, N>& result) {
  if (!value.isArray()) {
    return false;
  }
  result.reserve(value.size());
  for (const auto& sub_json : value) {
    if (sub_json.isNull()) {
      result.push_back(std::nullopt);
    } else {
      T t;
      if (!GetTaskValue(sub_json, name, t)) {
        return false;
      }
      result.push_back(std::move(t));
    }
  }
  return true;
}

template <class T, size_t N>
bool GetTaskValue(const Json::Value& value, const char* name,
                  absl::InlinedVector<T, N>& result) {
  if (!value.isArray()) {
    return false;
  }
  result.reserve(value.size());
  for (const auto& sub_json : value) {
    T t;
    if (!GetTaskValue(sub_json, name, t)) {
      return false;
    }
    result.push_back(std::move(t));
  }
  return true;
}

template <class T>
absl::StatusOr<T> GetTaskValue(const Json::Value& parent,
                               const std::string& context, const char* name) {
  Json::Value value = parent.get(name, Json::Value::null);
  if (value.isNull()) {
    return InvalidArgumentStrCat(context, " has no field '", name,
                                 "' in backend config");
  }
  T result;
  if (!GetTaskValue(value, name, result)) {
    return InvalidArgumentStrCat(context, " could not cast field '", name,
                                 "' to correct type in backend config");
  }
  return std::move(result);
}

template <class T>
absl::StatusOr<T> GetOptionalTaskValue(const Json::Value& parent,
                                       const std::string& context,
                                       const char* name, T dflt) {
  Json::Value value = parent.get(name, Json::Value::null);
  if (value.isNull()) {
    return std::move(dflt);
  }
  T result;
  if (!GetTaskValue(value, name, result)) {
    return InvalidArgumentStrCat(context, " could not cast field '", name,
                                 "' to correct type in backend config");
  }
  return std::move(result);
}

template <class T>
absl::StatusOr<std::optional<T>> GetOptionalTaskValue(
    const Json::Value& parent, const std::string& context, const char* name) {
  Json::Value value = parent.get(name, Json::Value::null);
  if (value.isNull()) {
    return std::nullopt;
  }
  T result;
  if (!GetTaskValue(value, name, result)) {
    return InvalidArgumentStrCat(context, " could not cast field '", name,
                                 "' to correct type in backend config");
  }
  return std::move(result);
}

absl::StatusOr<Json::Value> GetJsonValue(const char* data, size_t size);

}  // namespace xla

#endif