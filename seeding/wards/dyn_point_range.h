// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "parlay/internal/file_map.h"
#include "parlay/primitives.h"

namespace dyn {
template <class Point_>
struct PointRange {
  using Point = Point_;
  using T = typename Point::T;
  using parameters = typename Point::parameters;
  using byte = uint8_t;

  long dimension() const { return params.dims; }

  PointRange() : values(std::shared_ptr<byte[]>(nullptr, std::free)) { n = 0; }

  template <typename Seq>
  PointRange(const Seq& data, unsigned d)
      : values(std::shared_ptr<byte[]>(nullptr, std::free)), n(data.size()) {
    params = parameters(d);
    std::cout << "Data: detected " << n << " points with dimension " << d
              << std::endl;
    int num_bytes = params.num_bytes();
    aligned_bytes = 64 * ((num_bytes - 1) / 64 + 1);
    if (aligned_bytes != num_bytes)
      std::cout << "Aligning bytes to " << aligned_bytes << std::endl;

    const long align = 1l << 21;
    long total_bytes = (n + 1) * aligned_bytes;
    long total_bytes_aligned = ((total_bytes + align - 1) / align) * align;
    byte* ptr = (byte*)aligned_alloc(align, total_bytes_aligned);
    if (ptr == nullptr) {
      std::cerr << "aligned_alloc(" << align << ", " << total_bytes_aligned
                << ") failed" << std::endl;
      std::abort();
    }
#ifdef MADV_HUGEPAGE
    madvise(ptr, total_bytes_aligned, MADV_HUGEPAGE);
#endif
    values = std::shared_ptr<byte[]>(ptr, std::free);

    parlay::parallel_for(0, n, [&](size_t i) {
      byte* val = values.get() + i * aligned_bytes;
      Point::translate_point(val, data[i], params);
    });

    std::fill(values.get() + n * aligned_bytes, values.get() + (n + 1) * aligned_bytes, (T)0);
  }

  size_t size() const { return n; }

  unsigned int get_dims() const { return params.dims; }

  Point operator[](long i) const {
    return Point(values.get() + i * aligned_bytes, i, params);
  }

  Point operator[](long i) {
    return Point(values.get() + i * aligned_bytes, i, params);
  }

  void AddZero() { n++; }

  void centroid(long i, long j, size_t sz1, size_t sz2) {
    const double total_size = static_cast<double>(sz1) + static_cast<double>(sz2);
    T* p1 = reinterpret_cast<T*>(values.get() + i * aligned_bytes);
    T* p2 = reinterpret_cast<T*>(values.get() + j * aligned_bytes);
    for (int k = 0; k < params.dims; k++) {
      const double v =
          (static_cast<double>(p1[k]) * static_cast<double>(sz1) +
           static_cast<double>(p2[k]) * static_cast<double>(sz2)) /
          total_size;
      p1[k] = static_cast<T>(v);
    }
  }

  parameters params;

 private:
  std::shared_ptr<byte[]> values;
  long aligned_bytes;
  size_t n;
};

}  // namespace dyn
