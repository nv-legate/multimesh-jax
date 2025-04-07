/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/json_utils.h"

namespace xla {

absl::StatusOr<Json::Value> GetJsonValue(const char* data, size_t size) {
  Json::Reader reader;
  Json::Value result;
  const char* start = data;
  const char* end = start + size;
  if (!reader.parse(start, end, result)) {
    return InvalidArgumentStrCat("invalid JSON received: ",
                                 std::string{data, size});
  }
  return std::move(result);
}

bool GetTaskValue(const Json::Value& value, const char* name, bool& result) {
  if (!value.isBool()) {
    return false;
  }
  result = value.asBool();
  return true;
}

bool GetTaskValue(const Json::Value& value, const char* name, int& result) {
  if (!value.isInt()) {
    return false;
  }
  result = value.asInt();
  return true;
}

bool GetTaskValue(const Json::Value& value, const char* name, int64_t& result) {
  if (!value.isInt()) {
    return false;
  }
  result = value.asInt();
  return true;
}

bool GetTaskValue(const Json::Value& value, const char* name,
                  std::string& result) {
  if (!value.isString()) {
    return false;
  }
  result = value.asString();
  return true;
}

}  // namespace xla
