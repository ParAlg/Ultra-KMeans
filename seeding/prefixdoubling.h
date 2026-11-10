#pragma once
#include "kmeans_utils.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template <typename T, typename Range>
std::pair<parlay::sequence<uint32_t>, parlay::sequence<uint32_t>>
PrefixDoubling(const Range &points, uint32_t k, double alpha = 1.05) {
  size_t n = points.size();
  k = std::min(k, static_cast<uint32_t>(n));
  parlay::sequence<T> distances(n, std::numeric_limits<T>::max());
  parlay::sequence<bool> centers_set(n, false);

  parlay::sequence<uint32_t> centers(k);
  parlay::sequence<uint32_t> assignments(n, 0);
  T sum_squared_distances = 0;
  size_t selected_centers = 0;
  double batch = 1.0;
  for (size_t round_id = 0; selected_centers < k; round_id++) {
    parlay::sequence<uint32_t> new_centers;
    if (round_id == 0) {
      uint32_t random_center = parlay::hash32(round_id) % n;
      new_centers.push_back(random_center);
    } else {
      size_t remaining = k - selected_centers;
      size_t target = static_cast<size_t>(batch);
      size_t num_centers_to_add =
          std::max((size_t)1, std::min(target, remaining));
      new_centers = PickRandomCenters(
          distances, round_id, sum_squared_distances, num_centers_to_add);
      if (new_centers.empty()) {
        uint32_t best_j = n;
        T best_d = T(0);
        for (size_t j = 0; j < n; j++) {
          if (!centers_set[j] && distances[j] > best_d) {
            best_d = distances[j];
            best_j = static_cast<uint32_t>(j);
          }
        }
        if (best_j == n) {
          for (size_t j = 0; j < n; j++) {
            if (!centers_set[j]) {
              best_j = static_cast<uint32_t>(j);
              break;
            }
          }
        }
        if (best_j < n) new_centers.push_back(best_j);
      }
    }
    batch *= alpha;
    // resize if more than k centers are selected
    if (selected_centers + new_centers.size() > k) {
      new_centers.resize(k - selected_centers);
    }
    if (!new_centers.empty()) {
      parlay::parallel_for(0, new_centers.size(), [&](size_t i) {
        uint32_t c = new_centers[i];
        centers_set[c] = true;
        centers[selected_centers + i] = c;
      });
      sum_squared_distances = BatchUpdateDistances(
          points, distances, new_centers, assignments, selected_centers);
      selected_centers += new_centers.size();
    }
  }
  return {centers, assignments};
}
