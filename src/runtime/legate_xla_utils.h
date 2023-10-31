#include "legate.h"
#include "legate_xla_common.h"

namespace legate_xla {

legate::Type::Code SupportedTypeToLegateType(SupportedType type);

size_t SupportedTypeSizeOf(SupportedType type);

size_t ShapeNumElements(Shape shape);

} // namespace legate_xla
