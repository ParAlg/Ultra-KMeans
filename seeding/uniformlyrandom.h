#pragma once
#include <random>
#include <unordered_map>

#include "algorithms/utils/point_range.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template <typename T, typename Range>
parlay::sequence<uint32_t> UniformlyRandom(const Range &points, uint32_t k) {
  size_t n = points.size();
  k = std::min(k, static_cast<uint32_t>(n));

  std::unordered_map<uint32_t, uint32_t> virtual_arr;
  virtual_arr.reserve(k * 2);

  auto get = [&](uint32_t i) -> uint32_t {
    auto it = virtual_arr.find(i);
    return (it == virtual_arr.end()) ? i : it->second;
  };

  std::mt19937 rng(0);
  parlay::sequence<uint32_t> centers(k);
  for (uint32_t i = 0; i < k; i++) {
    std::uniform_int_distribution<uint32_t> dist(i, static_cast<uint32_t>(n) - 1);
    uint32_t j = dist(rng);
    centers[i] = get(j);
    virtual_arr[j] = get(i);
  }
  return centers;
}
