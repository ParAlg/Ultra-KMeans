#pragma once
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template <typename R, typename T>
uint32_t PickRandomCenter(const R &sq_dist, T cutoff,
                          const parlay::sequence<T> &weights = {}) {
  size_t n = sq_dist.size();
  for (size_t i = 0; i < n; i++) {
    T contrib = weights.empty() ? sq_dist[i] : sq_dist[i] * weights[i];
    if (contrib != 0) {
      cutoff -= contrib;
      if (cutoff <= 0) return static_cast<uint32_t>(i);
    }
  }
  for (size_t i = n; i-- > 0;) {
    T contrib = weights.empty() ? sq_dist[i] : sq_dist[i] * weights[i];
    if (contrib != 0) return static_cast<uint32_t>(i);
  }
  return static_cast<uint32_t>(n - 1);
}

template <typename T, typename Range>
T BatchUpdateDistances(const Range &points, parlay::sequence<T> &sq_dist,
                       const parlay::sequence<uint32_t> &new_center_ids,
                       parlay::sequence<uint32_t> &assignments,
                       size_t selected_so_far) {
  parlay::parallel_for(0, points.size(), [&](size_t i) {
    for (size_t j = 0; j < new_center_ids.size(); j++) {
      T d = points[i].distance(points[new_center_ids[j]]);
      if (d < sq_dist[i]) {
        sq_dist[i] = d;
        assignments[i] = static_cast<uint32_t>(selected_so_far + j);
      }
    }
  });
  return parlay::reduce(sq_dist);
}

template <typename T>
parlay::sequence<uint32_t> PickRandomCenters(
    const parlay::sequence<T> &sq_dist, size_t round_id,
    T phi, size_t num_centers_to_add) {
  size_t n = sq_dist.size();
  parlay::random_generator parallel_generator(round_id * n);
  auto pred = parlay::delayed_seq<bool>(n, [&](size_t i) {
    double p = static_cast<double>(num_centers_to_add) *
               static_cast<double>(sq_dist[i]) / static_cast<double>(phi);
    if (p >= 1.0) return true;
    if (p <= 0.0) return false;
    std::bernoulli_distribution bern(p);
    auto gen = parallel_generator[i];
    return bern(gen);
  });
  return parlay::pack_index<uint32_t>(pred);
}
