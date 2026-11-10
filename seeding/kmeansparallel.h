#pragma once
#include "kmeans_utils.h"
#include "kmeansplusplus.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

// k-means|| seeding (Bahmani et al., 2012).
template <typename T, typename Range>
std::pair<parlay::sequence<uint32_t>, parlay::sequence<uint32_t>>
ParallelPlusPlus(const Range &points, uint32_t k,
                 size_t num_rounds = 5,
                 double oversampling_factor = 1.0) {
  size_t n = points.size();
  k = std::min(k, static_cast<uint32_t>(n));

  const size_t l = std::max(size_t(1), static_cast<size_t>(oversampling_factor * k));

  parlay::sequence<T>        sq_dist(n, std::numeric_limits<T>::max());
  parlay::sequence<uint32_t> nearest(n, 0);

  uint32_t c0 = parlay::hash32(0) % static_cast<uint32_t>(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    sq_dist[i] = points[i].distance(points[c0]);
    nearest[i] = c0;
  });
  T phi = parlay::reduce(sq_dist);

  parlay::sequence<uint32_t> candidates;
  candidates.push_back(c0);

  for (size_t r = 1; r <= num_rounds && phi > T(0); r++) {
    auto new_centers = PickRandomCenters(sq_dist, r, phi, l);
    if (new_centers.empty()) continue;

    parlay::parallel_for(0, n, [&](size_t i) {
      for (uint32_t c : new_centers) {
        T d = points[i].distance(points[c]);
        if (d < sq_dist[i]) {
          sq_dist[i] = d;
          nearest[i] = c;
        }
      }
    });
    phi = parlay::reduce(sq_dist);

    for (uint32_t c : new_centers) candidates.push_back(c);
  }

  parlay::sequence<size_t> raw_counts(n, 0);
  parlay::parallel_for(0, n, [&](size_t i) {
    __sync_fetch_and_add(&raw_counts[nearest[i]], size_t(1));
  });
  parlay::sequence<T> weights(n, T(0));
  parlay::parallel_for(0, n, [&](size_t i) {
    weights[i] = static_cast<T>(raw_counts[i]);
  });

  auto [center_ids, _spp_assignments] = SequentialPlusPlus<T>(points, k, weights);
  (void)_spp_assignments;

  parlay::sequence<T>        assign_dist(n, std::numeric_limits<T>::max());
  parlay::sequence<uint32_t> assignments(n, 0);
  BatchUpdateDistances(points, assign_dist, center_ids, assignments, 0);

  return {center_ids, assignments};
}
