
#include "xla/pjrt/legate/legate_sharding.h"

#include "src/zuku/mesh.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/service/spmd/spmd_partitioner_util.h"

namespace xla {

// Returns whether the `sharding` is replicated. If the sharding
// has no assigned devices, returns true. Or if assigned
// devices and tile dimensions, returns true if all sharding
// dimensions are size 1.
bool IsReplicatedOnAssignedSubmesh(const HloSharding& sharding) {
  if (sharding.IsReplicated()) {
    return true;
  }
  if (sharding.ReplicateOnLastTileDim()) {
    int last_dim = sharding.tile_assignment().num_dimensions() - 1;
    for (int dim = 0; dim < last_dim; ++dim) {
      if (sharding.tile_assignment().dim(dim) != 1) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool IsReplicatedOrSubmeshReplicate(const HloSharding& sharding) {
  return sharding.IsReplicated() || IsReplicatedOnAssignedSubmesh(sharding) ||
         sharding.IsTileMaximal();
}

bool MeshEquivalentSharding(const HloSharding& lhs, const HloSharding& rhs) {
  if (IsReplicatedOrSubmeshReplicate(lhs)) {
    return IsReplicatedOrSubmeshReplicate(rhs);
  }

  if (IsReplicatedOrSubmeshReplicate(rhs)) {
    return false;
  }

  return lhs.tile_assignment().dimensions() ==
         rhs.tile_assignment().dimensions();
}

zuku::DeviceList GetDevices(const HloSharding& sharding,
                            zuku::DeviceList default_devices) {
  if (sharding.IsReplicated()) {
    return default_devices;
  }
  return zuku::DeviceList{
      {.start = sharding.tile_assignment().first(),
       .num_devices = sharding.tile_assignment().num_elements()}};
}

Shape GetSpmdShape(const HloInstruction* instruction) {
  if (!instruction->has_sharding()) {
    return instruction->shape();
  }

  if (!instruction->shape().IsTuple()) {
    return spmd::MakePartitionedShape(instruction->shape(),
                                      instruction->sharding());
  }

  ShapeProto shape;
  shape.set_element_type(PrimitiveType::TUPLE);

  for (int64_t idx = 0; idx < instruction->shape().tuple_shapes_size(); ++idx) {
    Shape spmd_shape = spmd::MakePartitionedShape(
        instruction->shape().tuple_shapes(idx),
        instruction->sharding().tuple_elements()[idx]);
    *shape.add_tuple_shapes() = spmd_shape.ToProto();
  }
  return Shape{shape};
}

template <class T>
std::ostream& operator<<(std::ostream& os, const IndexVector<T>& vec) {
  os << "{ ";
  for (const auto& t : vec) {
    os << t << ",";
  }
  os << " }";
  return os;
}

// Increase or decrease the size of a sharding to match a tile assignment with a
// given number of elements Returns the new sharding if heuristics can find a
// valid resharding, other nullopt
std::optional<HloSharding> ResizeSharding(
    const Shape& shape, const HloSharding& sharding, int64_t num_elements,
    std::optional<zuku::DeviceList> devices) {
  const auto& iota = sharding.tile_assignment().iota();
  if (iota.has_value()) {
    for (int dim = 0; dim < iota->transpose_perm().size(); ++dim) {
      if (dim != iota->transpose_perm()[dim]) {
        // cannot reshard with a transpose permutation
        return std::nullopt;
      }
    }
  }

  // we have to guess at a sharding for this, if we can
  // grow or shrink the lead sharding dimension
  OpSharding guess_sharding;
  guess_sharding.set_type(OpSharding::OTHER);
  const int64_t src_size = sharding.tile_assignment().num_elements();
  const int64_t dst_size = num_elements;

  const int64_t last_dim = sharding.tile_assignment().num_dimensions() - 1;
  bool matched = false;
  for (size_t dim = 0; dim < sharding.tile_assignment().num_dimensions();
       ++dim) {
    const int64_t old_dim_size = sharding.tile_assignment().dim(dim);
    const int64_t new_dim_size = old_dim_size * dst_size / src_size;
    const int64_t shape_dim = shape.dimensions(dim);
    if (!matched && old_dim_size > 1 && new_dim_size > 0 &&
        new_dim_size <= shape_dim) {
      // try to use this dimension to grow or shrink the sharding
      if (src_size > dst_size && src_size % dst_size) {
        continue;
      } else if (src_size < dst_size && dst_size % src_size) {
        continue;
      }

      guess_sharding.add_tile_assignment_dimensions(new_dim_size);
      matched = true;
    } else {
      guess_sharding.add_tile_assignment_dimensions(old_dim_size);
    }
    if (dim == last_dim && sharding.ReplicateOnLastTileDim() &&
        guess_sharding.tile_assignment_dimensions(dim) > 1) {
      guess_sharding.set_replicate_on_last_tile_dim(true);
    }
  }
  if (matched) {
    if (sharding.tile_assignment().iota().has_value() || !devices.has_value()) {
      guess_sharding.mutable_iota_reshape_dims()->Add(num_elements);
      guess_sharding.mutable_iota_transpose_perm()->Add(0);
      if (devices.has_value()) {
        guess_sharding.set_iota_offset(devices->start());
      }
    } else {
      guess_sharding.mutable_tile_assignment_devices()->Assign(devices->begin(),
                                                               devices->end());
    }
    auto new_sharding = HloSharding::FromProto(guess_sharding);
    if (new_sharding.ok()) {
      return *std::move(new_sharding);
    }
  }

  return std::nullopt;
}

absl::StatusOr<OpSharding> LogicalToPhysicalSharding(
    const IndexVector<IndexVector<int>>& logical_to_device_axes,
    std::pair<int64_t, int64_t> slice, const IndexVector<int64_t>& dims,
    bool list_all_devices) {
  return LogicalToPhysicalSharding(
      logical_to_device_axes,
      zuku::DeviceList::Create(slice.first, slice.second), dims,
      list_all_devices);
}

// The computation here is annoyingly complicated. There are 0..n device axes
// assigned to each logical axis. This defines a logical device grid. Consider
// the logical->device mapping [ (0,2), (), (1,) ] The 1st logical axes is
// sharded over device axes 0,2, the 2nd logical axis is not sharded, the 3rd
// logical axis is sharded over device axis 1. For a device grid [2,2,2], the
// logical tile dims are then tile_dims = [4,1,2]. To compute the device
// assignment requires some tedious index math. Logical device 5 -> Logical
// Index [2,0,1] -> Device MultiIndex [ (1,0), (), (1) ]
// -> Device Index [1,1,0] -> Physical device 6
absl::StatusOr<OpSharding> LogicalToPhysicalSharding(
    const IndexVector<IndexVector<int>>& logical_to_device_axes,
    const zuku::DeviceList& devices, const IndexVector<int64_t>& dims,
    bool list_all_devices) {
  size_t num_logical_dims = logical_to_device_axes.size();
  size_t num_device_dims = dims.size();

  // figure out which device dims are not used
  IndexVector<int> device_axes_used(dims.size(), 0);
  bool fully_replicated = true;
  for (const auto& axes : logical_to_device_axes) {
    for (auto dim : axes) {
      device_axes_used[dim] = 1;
      if (dims[dim] > 1) {
        fully_replicated = false;
      }
    }
  }

  // if needing to list all devices, make all the device axes 1
  // and explicitly write out all devices as the replicated last dim
  if (logical_to_device_axes.empty() ||
      (fully_replicated && !list_all_devices)) {
    OpSharding sharding;
    sharding.set_type(OpSharding::REPLICATED);
    return sharding;
  }

  IndexVector<int> replicated_device_axes;
  int64_t replication_size = 1;
  int64_t total_mesh_size = 1;
  for (size_t dim = 0; dim < dims.size(); ++dim) {
    if (device_axes_used[dim] == 0) {
      replicated_device_axes.push_back(dim);
      replication_size *= dims[dim];
    }
    total_mesh_size *= dims[dim];
  }

  IndexVector<int64_t> tile_dims(num_logical_dims);
  IndexVector<int64_t> dim_permutation;
  dim_permutation.reserve(dims.size());
  for (size_t dim = 0; dim < num_logical_dims; ++dim) {
    IndexVector<int> assigned_device_axes = logical_to_device_axes[dim];
    std::sort(assigned_device_axes.begin(), assigned_device_axes.end());
    int64_t tile_dim = 1;
    for (int device_ax : assigned_device_axes) {
      tile_dim *= dims[device_ax];
      dim_permutation.push_back(device_ax);
    }
    tile_dims[dim] = tile_dim;
  }

  // all unused device axes go at the end
  for (auto dim = 0; dim < dims.size(); ++dim) {
    if (!device_axes_used[dim]) {
      dim_permutation.push_back(dim);
    }
  }

  if (!replicated_device_axes.empty()) {
    tile_dims.push_back(replication_size);
  }

  OpSharding sharding;
  sharding.set_type(OpSharding::OTHER);
  sharding.mutable_tile_assignment_dimensions()->Add(tile_dims.begin(),
                                                     tile_dims.end());

  bool need_permutation = false;
  for (auto idx = 0; idx < dims.size(); ++idx) {
    if (idx != dim_permutation[idx]) {
      need_permutation = true;
      break;
    }
  }

  if (need_permutation) {
    sharding.mutable_iota_transpose_perm()->Add(dim_permutation.begin(),
                                                dim_permutation.end());
    sharding.mutable_iota_reshape_dims()->Add(dims.begin(), dims.end());
  } else {
    sharding.mutable_iota_reshape_dims()->Add(total_mesh_size);
    sharding.mutable_iota_transpose_perm()->Add(0);
  }

  sharding.set_replicate_on_last_tile_dim(!replicated_device_axes.empty());
  sharding.set_iota_offset(devices[0]);

  return sharding;
}

absl::StatusOr<zuku::ShardedShape> CanonicalizeSharding(
    const HloSharding& sharding, const TileAssignment& ta,
    const IotaTileAssignment& iota, const Shape& shape, bool shape_is_global) {
  absl::InlinedVector<int64_t, 6> full_permutation;
  int64_t iota_dim = 0;
  int64_t spanned_dimension = 1;
  absl::InlinedVector<absl::InlinedVector<int64_t, 6>, 6> dimension_groups;
  dimension_groups.emplace_back();

  absl::InlinedVector<int64_t, 6> permuted_reshape_dims;
  if (iota.transpose_perm().empty()) {
    permuted_reshape_dims.insert(permuted_reshape_dims.end(),
                                 iota.reshape_dims().begin(),
                                 iota.reshape_dims().end());
  } else {
    permuted_reshape_dims.resize(iota.reshape_dims().size());
    for (int64_t idx = 0; idx < iota.transpose_perm().size(); ++idx) {
      const int64_t iota_permuted_dim = iota.transpose_perm()[idx];
      permuted_reshape_dims[idx] = iota.reshape_dims()[iota_permuted_dim];
    }
  }

  // the iota reshape might group together multiple tile dimensions
  // ungroup them now
  int64_t ta_dim = 0;
  for (;
       ta_dim < ta.num_dimensions() && iota_dim < permuted_reshape_dims.size();
       ++ta_dim) {
    spanned_dimension *= ta.dim(ta_dim);
    const int64_t iota_group_span = permuted_reshape_dims[iota_dim];
    dimension_groups.back().push_back(ta_dim);
    if (spanned_dimension >= iota_group_span) {
      ++iota_dim;
      if (iota_dim < permuted_reshape_dims.size()) {
        dimension_groups.emplace_back();
      }
      spanned_dimension = 1;
    }
  }
  for (; ta_dim < ta.num_dimensions(); ++ta_dim) {
    dimension_groups.back().emplace_back(ta_dim);
  }

  absl::InlinedVector<absl::InlinedVector<int64_t, 6>, 6> permuted_groups(
      dimension_groups.size());
  int64_t permuted_dim = 0;
  for (int64_t src_dim : iota.transpose_perm()) {
    permuted_groups[permuted_dim++] = std::move(dimension_groups[src_dim]);
  }

  full_permutation.reserve(ta.num_dimensions() + 1);
  for (auto&& group : permuted_groups) {
    for (int64_t dim : group) {
      full_permutation.push_back(dim);
    }
  }

  const int64_t num_shape_dims = sharding.ReplicateOnLastTileDim()
                                     ? ta.num_dimensions() - 1
                                     : ta.num_dimensions();

  std::vector<zuku::ShardingDim> dims(ta.num_dimensions());
  int64_t shape_dim = 0;
  int64_t perm_dim = 0;
  std::optional<int64_t> replicated_dim{std::nullopt};
  for (int64_t dim = 0; dim < full_permutation.size(); ++dim) {
    VLOG(5) << sharding << " permutes " << dim << " -> "
            << full_permutation[dim];

    const int64_t extent = ta.dim(full_permutation[dim]);

    const auto [size, sharding] = [&]() -> std::pair<int64_t, int64_t> {
      if (full_permutation[dim] == num_shape_dims) {
        return {1, extent};
      }
      int64_t dim_size = shape.dimensions(full_permutation[dim]);
      if (!shape_is_global) {
        dim_size *= extent;
      }
      return {dim_size, extent};
    }();

    zuku::ShardingDim next{
        .size = size, .sharding = sharding, .permutation = perm_dim++};
    dims[full_permutation[dim]] = std::move(next);
  }

  zuku::DeviceList devices{
      {.start = iota.offset(), .num_devices = iota.num_elements()}};

  zuku::SupportedType zuku_type;

  switch (shape.element_type()) {
    case PrimitiveType::PRED:
      zuku_type = zuku::SupportedType::PRED;
      break;
    case PrimitiveType::F16:
      zuku_type = zuku::SupportedType::F16;
      break;
    case PrimitiveType::BF16:
      zuku_type = zuku::SupportedType::BF16;
      break;
    case PrimitiveType::F32:
      zuku_type = zuku::SupportedType::F32;
      break;
    case PrimitiveType::F64:
      zuku_type = zuku::SupportedType::F64;
      break;
    case PrimitiveType::S8:
      zuku_type = zuku::SupportedType::S8;
      break;
    case PrimitiveType::S16:
      zuku_type = zuku::SupportedType::S16;
      break;
    case PrimitiveType::S32:
      zuku_type = zuku::SupportedType::S32;
      break;
    case PrimitiveType::S64:
      zuku_type = zuku::SupportedType::S64;
      break;
    case PrimitiveType::U8:
      zuku_type = zuku::SupportedType::U8;
      break;
    case PrimitiveType::U16:
      zuku_type = zuku::SupportedType::U16;
      break;
    case PrimitiveType::U32:
      zuku_type = zuku::SupportedType::U32;
      break;
    case PrimitiveType::U64:
      zuku_type = zuku::SupportedType::U64;
      break;
    case PrimitiveType::C64:
      zuku_type = zuku::SupportedType::C64;
      break;
    case PrimitiveType::C128:
      zuku_type = zuku::SupportedType::C128;
      break;
    default:
      return Internal("Unsupported type: %s",
                      PrimitiveType_Name(shape.element_type()));
  }

  return zuku::ShardedShape{.type = zuku_type,
                            .sharding = {
                                .dims = std::move(dims),
                                .devices = std::move(devices),
                            }};
}

absl::StatusOr<zuku::ShardedShape> CanonicalizeSharding(
    const HloSharding& sharding, const Shape& shape, bool shape_is_global) {
  auto&& ta = sharding.tile_assignment();

  if (ta.iota().has_value()) {
    return CanonicalizeSharding(sharding, ta, *ta.iota(), shape,
                                shape_is_global);
  }

  // see if this is actually an iota sharding, but not structured as one
  int64_t prev = ta.first() - 1;
  for (int64_t dev : ta.array()) {
    if (dev != (prev + 1)) {
      return InvalidArgumentStrCat(
          "CanonicalizeSharding: only supported for iota tile assignment");
    }
    prev = dev;
  }

  auto iota = IotaTileAssignment::Create(ta.dimensions(), {ta.num_elements()},
                                         {0}, ta.first());
  return CanonicalizeSharding(sharding, ta, iota, shape, shape_is_global);
}

absl::StatusOr<HloSharding> ReplicateDims(
    absl::Span<const int64_t> replicated_dims, const HloSharding& sharding) {
  OpSharding op_sharding;

  op_sharding.set_type(OpSharding::OTHER);

  auto&& ta = sharding.tile_assignment();
  if (!ta.iota().has_value()) {
    return InvalidArgumentStrCat(
        "ReplicateDims only supported for iota tile assignment");
  }
  std::vector<int64_t> current_permutation;
  auto&& iota = *ta.iota();
  int64_t replication = 1;
  if (iota.transpose_perm().empty()) {
    current_permutation.resize(ta.num_dimensions());
    std::iota(current_permutation.begin(), current_permutation.end(), 0);
  } else {
    int64_t iota_dim = 0;
    int64_t spanned_dimension = 1;
    std::vector<std::vector<int64_t>> dimension_groups;
    dimension_groups.emplace_back();

    // the iota reshape might group together multiple dimensions
    // ungroup them now
    int64_t ta_dim = 0;
    for (; ta_dim < ta.num_dimensions() &&
           iota_dim < iota.transpose_perm().size();
         ++ta_dim) {
      spanned_dimension *= ta.dim(ta_dim);
      const int64_t iota_permuted_dim = iota.transpose_perm()[iota_dim];
      const int64_t index_group_span = iota.reshape_dims()[iota_permuted_dim];
      dimension_groups.back().push_back(ta_dim);
      if (spanned_dimension >= index_group_span) {
        ++iota_dim;
        if (iota_dim < iota.transpose_perm().size()) {
          dimension_groups.emplace_back();
        }
        spanned_dimension = 1;
      }
    }
    for (; ta_dim < ta.num_dimensions(); ++ta_dim) {
      dimension_groups.back().emplace_back(ta_dim);
    }

    std::vector<std::vector<int64_t>> permuted_groups(dimension_groups.size());
    int64_t permuted_dim = 0;
    for (int64_t src_dim : iota.transpose_perm()) {
      permuted_groups[permuted_dim++] = dimension_groups[src_dim];
    }

    current_permutation.reserve(ta.num_dimensions() + 1);
    for (auto&& group : permuted_groups) {
      for (int64_t dim : group) {
        current_permutation.push_back(dim);
      }
    }
  }

  std::vector<int64_t> new_reshape_dims;
  new_reshape_dims.reserve(current_permutation.size() + replicated_dims.size());
  for (const int64_t dim : current_permutation) {
    new_reshape_dims.push_back(ta.dim(dim));
  }
  for (const int64_t dim : replicated_dims) {
    new_reshape_dims.push_back(1);
  }

  const int64_t num_shape_dims = sharding.ReplicateOnLastTileDim()
                                     ? ta.num_dimensions() - 1
                                     : ta.num_dimensions();

  std::vector<int64_t> new_permutation;
  new_permutation.reserve(current_permutation.size() + replicated_dims.size());
  for (const int64_t dim : current_permutation) {
    new_permutation.push_back(dim);
  }
  int64_t extra_dim = new_permutation.size();
  for (const int64_t dim : replicated_dims) {
    new_permutation.push_back(current_permutation[dim]);
    new_permutation[dim] = extra_dim++;
  }

  const int64_t total_replication = [&] {
    int64_t replication = 1;
    if (sharding.ReplicateOnLastTileDim()) {
      replication *= ta.dim(num_shape_dims);
    }
    for (const int64_t dim : replicated_dims) {
      replication *= ta.dim(dim);
    }
    return replication;
  }();

  if (total_replication == ta.num_elements()) {
    // treat as replicated
    return HloSharding::Replicate();
  }

  std::vector<int64_t> ta_dimensions;
  ta_dimensions.reserve(num_shape_dims + 1);
  ta_dimensions.insert(ta_dimensions.end(), ta.dimensions().begin(),
                       ta.dimensions().end());
  if (sharding.ReplicateOnLastTileDim()) {
    ta_dimensions[num_shape_dims] = total_replication;
  } else {
    ta_dimensions.push_back(total_replication);
  }

  for (const int64_t dim : replicated_dims) {
    ta_dimensions[dim] = 1;
  }

  op_sharding.mutable_tile_assignment_dimensions()->Add(ta_dimensions.begin(),
                                                        ta_dimensions.end());
  op_sharding.mutable_iota_reshape_dims()->Add(new_reshape_dims.begin(),
                                               new_reshape_dims.end());
  op_sharding.mutable_iota_transpose_perm()->Add(new_permutation.begin(),
                                                 new_permutation.end());

  op_sharding.set_replicate_on_last_tile_dim(true);

  return HloSharding::FromProto(op_sharding);
}

absl::StatusOr<zuku::ShardedShape> XlaShapeToLegateShape(
    const Shape& shape, const zuku::DeviceList& devices,
    const HloSharding& sharding, bool shape_is_global) {
  zuku::SupportedType zuku_type;
  switch (shape.element_type()) {
    case PrimitiveType::PRED:
      zuku_type = zuku::SupportedType::PRED;
      break;
    case PrimitiveType::F16:
      zuku_type = zuku::SupportedType::F16;
      break;
    case PrimitiveType::BF16:
      zuku_type = zuku::SupportedType::BF16;
      break;
    case PrimitiveType::F32:
      zuku_type = zuku::SupportedType::F32;
      break;
    case PrimitiveType::F64:
      zuku_type = zuku::SupportedType::F64;
      break;
    case PrimitiveType::S8:
      zuku_type = zuku::SupportedType::S8;
      break;
    case PrimitiveType::S16:
      zuku_type = zuku::SupportedType::S16;
      break;
    case PrimitiveType::S32:
      zuku_type = zuku::SupportedType::S32;
      break;
    case PrimitiveType::S64:
      zuku_type = zuku::SupportedType::S64;
      break;
    case PrimitiveType::U8:
      zuku_type = zuku::SupportedType::U8;
      break;
    case PrimitiveType::U16:
      zuku_type = zuku::SupportedType::U16;
      break;
    case PrimitiveType::U32:
      zuku_type = zuku::SupportedType::U32;
      break;
    case PrimitiveType::U64:
      zuku_type = zuku::SupportedType::U64;
      break;
    case PrimitiveType::C64:
      zuku_type = zuku::SupportedType::C64;
      break;
    case PrimitiveType::C128:
      zuku_type = zuku::SupportedType::C128;
      break;
    default:
      return Internal("Unsupported type: %s",
                      PrimitiveType_Name(shape.element_type()));
  }

  if (sharding.IsReplicated() || sharding.IsTileMaximal()) {
    std::vector<zuku::ShardingDim> sharding_dims;
    sharding_dims.reserve(shape.dimensions_size());
    int64_t dim_number = 0;
    for (auto&& dim : shape.dimensions()) {
      sharding_dims.push_back(
          {.size = dim, .sharding = 1, .permutation = dim_number++});
    }
    zuku::ShardedShape sharded_shape{
        .type = zuku_type,
        .sharding = {.dims = std::move(sharding_dims), .devices = devices}};
    return sharded_shape;
  }

  TF_ASSIGN_OR_RETURN(zuku::ShardedShape sharded_shape,
                      CanonicalizeSharding(sharding, shape, shape_is_global));

  if (devices != sharded_shape.sharding.devices) {
    if (devices.size() != sharded_shape.sharding.devices.size()) {
      return InvalidArgumentStrCat(
          "devices=[", devices.start(), "...", devices.stop(),
          ") given to XlaShapeToLegateShape differs from sharding: ",
          sharding.ToString());
    }
    // we can reshard a cross device meshes with the same shap
    sharded_shape.sharding.devices = devices;
  }

  return sharded_shape;
}

}  // namespace xla