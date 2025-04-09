#include "xla/pjrt/legate/color_dfs.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/mpmd_instruction.h"

namespace xla {


std::vector<HloInstruction*> ColorSortedPostorder(HloComputation* computation)
{
  std::vector<HloInstruction*> to_visit = { computation->root_instruction() };
  std::vector<HloInstruction*> dfs;
  dfs.reserve(computation->instruction_count());
  absl::flat_hash_map<HloInstruction*,bool> visited;

  absl::flat_hash_map<std::string,int64_t> color_priority;
  std::optional<std::string> last_color;
  int64_t next_priority = 1;

  while (!to_visit.empty()){
    auto* next = to_visit.back();
    VLOG(5) << "next " << next->name() << " in DFS color sort";
    if (!visited.contains(next)){
      absl::InlinedVector<HloInstruction*,2> operands = next->mutable_operands();
      auto color = Color(next);
      if (color.has_value() && color != last_color){
        color_priority[*color] = next_priority++;
        last_color = color;
      }
      std::stable_sort(operands.begin(), operands.end(), [&](HloInstruction* lhs, HloInstruction* rhs){
        auto lhs_color = Color(lhs);
        auto rhs_color = Color(rhs);
        if (lhs_color.has_value() && rhs_color.has_value()){
          // higher priority means first in the operand list (and later in the DFS)
          return color_priority[*lhs_color] > color_priority[*rhs_color];
        }
        // anything with a color should be moved first in the operands (and later in the DFS)
        if (lhs_color.has_value()){
          return true;
        }
        return false;
      });
      for (auto* operand : operands){
        VLOG(5) << "   color=" << Color(operand).value_or("none") << " " << operand->name() << " is next operand";
        to_visit.push_back(operand);
      }
      visited[next] = false;
      continue;
    }
    to_visit.pop_back();
    if (visited[next]){
      continue;
    }

    if (next->opcode() == HloOpcode::kGetTupleElement && next->operand(0)->opcode() == HloOpcode::kCustomCall){
      // put all the get-tuple-elements together, this is safe to do
      for (auto* user : next->mutable_operand(0)->users()){
        dfs.push_back(user);
        visited[user] = true;
      }
    } else {
      visited[next] = true;
      dfs.push_back(next);
    }
  }
  return dfs;
}

} // namespace xla
