#ifndef TINYINFER_MARLIN_COMPAT_H
#define TINYINFER_MARLIN_COMPAT_H
// The Marlin sources came from vLLM's quantization/marlin and used its
// STD_TORCH_CHECK for construction-time validation. This tree has no torch headers
// (kernels are built with bare nvcc into cubins), so the same check is expressed with
// the standard library. Every use in scalar_type.hpp validates compile-time constants,
// so a failure here means the type table is wrong, not that input data is bad.
#include <stdexcept>
#include <string>

#define STD_TORCH_CHECK(cond, ...)                                     \
  do {                                                                 \
    if (!(cond)) {                                                     \
      throw std::runtime_error(std::string("marlin: ") + #cond);       \
    }                                                                  \
  } while (0)

#endif
