#pragma once
#include <random>

#include "algorithms/utils/point_range.h"
#include "kmeans_utils.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template <typename T, typename Range>
T UpdateDistances(const Range &points, parlay::sequence<T> &sq_dist,
                  uint32_t new_center_id, uint32_t new_center_idx,
                  parlay::sequence<uint32_t> &assignments,
                  const parlay::sequence<T> &weights = parlay::sequence<T>{}) {
  const bool track = !assignments.empty();
  parlay::parallel_for(0, points.size(), [&](size_t i) {
    T d = points[i].distance(points[new_center_id]);
    if (d < sq_dist[i]) {
      sq_dist[i] = d;
      if (track) assignments[i] = new_center_idx;
    }
  });
  if (weights.empty()) {
    return parlay::reduce(sq_dist);
  }
  auto weighted = parlay::delayed_seq<T>(
      points.size(), [&](size_t i) { return sq_dist[i] * weights[i]; });
  return parlay::reduce(weighted);
}

// k-means++ seeding (Arthur & Vassilvitskii, 2007).
template <typename T, typename Range>
std::pair<parlay::sequence<uint32_t>, parlay::sequence<uint32_t>>
SequentialPlusPlus(
    const Range &points, uint32_t k,
    const parlay::sequence<T> &weights = parlay::sequence<T>{}) {
  size_t n = points.size();

  std::mt19937 generator(0);
  parlay::sequence<T> sq_dist(n, std::numeric_limits<T>::max());

  parlay::sequence<uint32_t> centers(k);
  parlay::sequence<uint32_t> assignments(n, 0);
  T phi = 0;

  for (size_t i = 0; i < k; i++) {
    uint32_t random_center;
    if (i == 0) {
      if (!weights.empty()) {
        T total_weight = parlay::reduce(weights);
        std::uniform_real_distribution<T> dist_w(T(0), total_weight);
        random_center = PickRandomCenter(weights, dist_w(generator));
      } else {
        std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(n) - 1);
        random_center = dist(generator);
      }
    } else {
      std::uniform_real_distribution<T> dist(T(0), phi);
      T cutoff = dist(generator);
      random_center = PickRandomCenter(sq_dist, cutoff, weights);
    }
    centers[i] = random_center;
    phi = UpdateDistances(points, sq_dist, random_center,
                          static_cast<uint32_t>(i), assignments, weights);
  }
  return {centers, assignments};
}
