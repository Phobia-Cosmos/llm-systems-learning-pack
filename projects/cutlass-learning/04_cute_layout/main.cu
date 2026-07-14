#include <cute/layout.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/util/print.hpp>
#include <cute/util/print_tensor.hpp>

#include <cstdio>
#include <vector>

using namespace cute;

template <class Layout>
void print_mapping(char const* name, Layout const& layout) {
  std::printf("\n%s\n", name);
  std::printf("shape:stride = ");
  print(layout);
  std::printf("\ncoordinate -> linear index\n");
  for (int row = 0; row < size<0>(layout); ++row) {
    for (int column = 0; column < size<1>(layout); ++column) {
      std::printf("(%d,%d)->%2d  ", row, column, int(layout(row, column)));
    }
    std::printf("\n");
  }
}

template <class Layout>
bool has_unique_in_range_mapping(Layout const& layout) {
  std::vector<bool> seen(static_cast<std::size_t>(cosize(layout)), false);
  for (int row = 0; row < size<0>(layout); ++row) {
    for (int column = 0; column < size<1>(layout); ++column) {
      int const index = int(layout(row, column));
      if (index < 0 || index >= int(seen.size()) || seen[index]) {
        return false;
      }
      seen[index] = true;
    }
  }
  return true;
}

int main() {
  auto const row_major =
      make_layout(make_shape(Int<4>{}, Int<8>{}), LayoutRight{});
  auto const column_major =
      make_layout(make_shape(Int<4>{}, Int<8>{}), LayoutLeft{});
  auto const padded_row_major = make_layout(
      make_shape(Int<4>{}, Int<8>{}),
      make_stride(Int<9>{}, Int<1>{}));
  auto const hierarchical = make_layout(
      make_shape(Int<2>{}, make_shape(Int<2>{}, Int<2>{})),
      make_stride(Int<4>{}, make_stride(Int<2>{}, Int<1>{})));

  print_mapping("row-major 4x8", row_major);
  print_mapping("column-major 4x8", column_major);
  print_mapping("row-major 4x8 with one padding element per row",
                padded_row_major);
  print_mapping("hierarchical shape (2,(2,2))", hierarchical);

  std::printf("\nCuTe table view of the hierarchical layout\n");
  print_layout(hierarchical);

  auto const compact = coalesce(hierarchical);
  std::printf("\ncoalesce((2,(2,2)):(4,(2,1))) = ");
  print(compact);
  std::printf("\n");

  bool const passed = has_unique_in_range_mapping(row_major) &&
                      has_unique_in_range_mapping(column_major) &&
                      has_unique_in_range_mapping(padded_row_major) &&
                      has_unique_in_range_mapping(hierarchical) &&
                      int(row_major(2, 3)) == 19 &&
                      int(column_major(2, 3)) == 14 &&
                      int(padded_row_major(2, 3)) == 21 &&
                      int(hierarchical(0, 2)) == 1;
  std::printf("mapping uniqueness and known-coordinate checks: %s\n",
              passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
