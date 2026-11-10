/*
  Ward's Agglomerative Clustering for seeding k-means
  1. Exact Implementation:
    Input: Points, k
      - k: Number of final clusters
    Running Time: O(n^2 * polylog(n))
  2. Approximate Implementation
    Input: Points, k, eps, delta
      - k: Number of final clusters
      - eps: (1+eps) merges performed
      - delta: clusters of size within (1+delta) grouped together
    Approximation Factor: (1+eps)*(1+delta)
    Running Time: O(n * polylog(n))
*/

#pragma once
#include <cstdlib>
#include <tuple>

#include "kmeans_utils.h"
#include "lloyds/pairwise.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "wards/dyn_point_range.h"
#include "wards/wards_hac.h"

template <typename PointTy, typename Range>
auto WardsD2Sample(const Range &points, size_t num_samples, size_t num_rounds,
                   uint32_t seed, BuildParams BP) {
  using T = typename PointTy::distanceType;
  const size_t n = points.size();
  const unsigned d = points.get_dims();
  const size_t l =
      (num_samples + num_rounds - 1) / std::max<size_t>(1, num_rounds);

  const uint32_t c0 = parlay::hash32(seed) % static_cast<uint32_t>(n);
  auto sq_dist = parlay::sequence<T>::uninitialized(n);
  auto point_to_sample = parlay::sequence<uint32_t>::uninitialized(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    sq_dist[i] = points[i].distance(points[c0]);
    point_to_sample[i] = 0;
  });
  T phi = parlay::reduce(sq_dist);

  parlay::sequence<uint32_t> candidates;
  candidates.reserve(1 + num_rounds * l);
  candidates.push_back(c0);

  for (size_t r = 1; r <= num_rounds && phi > T(0); r++) {
    auto new_centers =
        PickRandomCenters(sq_dist, static_cast<size_t>(seed) + r, phi, l);
    if (new_centers.empty()) continue;
    const size_t offset = candidates.size();
    for (uint32_t c : new_centers) candidates.push_back(c);

    auto cand_seq = parlay::delayed_seq<PointTy>(
        new_centers.size(), [&](size_t i) { return points[new_centers[i]]; });
    auto cand_pr = dyn::PointRange<PointTy>(cand_seq, d);
    Graph<uint32_t> G(BP.R, new_centers.size());
    knn_index<dyn::PointRange<PointTy>, dyn::PointRange<PointTy>, uint32_t> I(
        BP);
    stats<uint32_t> BS(G.size());
    I.build_index(G, cand_pr, cand_pr, BS);
    QueryParams QP(1, BP.L, 1.35, (long)G.size(), (long)G.max_degree());
    parlay::parallel_for(0, n, [&](size_t i) {
      parlay::sequence<uint32_t> sp = {I.get_start()};
      auto pairElts =
          beam_search<PointTy, dyn::PointRange<PointTy>, uint32_t>(
              points[i], G, cand_pr, sp, QP)
              .first;
      auto &beamElts = pairElts.first;
      if (!beamElts.empty() && beamElts[0].second < sq_dist[i]) {
        sq_dist[i] = beamElts[0].second;
        point_to_sample[i] = static_cast<uint32_t>(offset + beamElts[0].first);
      }
    });
    phi = parlay::reduce(sq_dist);
  }
  return std::make_pair(std::move(candidates), std::move(point_to_sample));
}

template <typename PointTy, typename DistTy = typename PointTy::distanceType,
          typename RangeP, typename Range>
auto compute_weights_anns(RangeP &points, Range &sampled_points, BuildParams BP,
                          bool weighted,
                          parlay::sequence<uint32_t> warm_pts = {}) {
  const size_t n = points.size();
  const size_t ns = sampled_points.size();
  const size_t d = points.get_dims();

  parlay::sequence<uint32_t> point_to_sample;
  if (warm_pts.size() == n) {
    point_to_sample = std::move(warm_pts);
  } else {
    Graph<uint32_t> G(BP.R, ns);
    knn_index<Range, Range, uint32_t> I(BP);
    stats<uint32_t> BS(G.size());
    I.build_index(G, sampled_points, sampled_points, BS);
    QueryParams QP(1, BP.L, 1.35, (long)G.size(), (long)G.max_degree());
    point_to_sample = parlay::sequence<uint32_t>::uninitialized(n);
    parlay::parallel_for(0, n, [&](size_t i) {
      parlay::sequence<uint32_t> sp = {I.get_start()};
      auto pairElts = beam_search<PointTy, Range, uint32_t>(
                          points[i], G, sampled_points, sp, QP)
                          .first;
      point_to_sample[i] = pairElts.first[0].first;
    });
  }

  parlay::sequence<uint32_t> weights;
  if (weighted) {
    auto cluster_ids =
        parlay::sequence<std::tuple<uint32_t, uint32_t>>::uninitialized(n);
    parlay::parallel_for(0, n, [&](size_t i) {
      cluster_ids[i] =
          std::make_tuple(point_to_sample[i], static_cast<uint32_t>(i));
    });
    auto groups = parlay::group_by_index(cluster_ids, ns);
    parlay::parallel_for(0, ns, [&](size_t i) {
      if (groups[i].size() == 0) return;
      auto subset = parlay::delayed_seq<PointTy>(
          groups[i].size(), [&](size_t j) { return points[groups[i][j]]; });
      compute_mean<PointTy>(subset, sampled_points, i, d);
    });
    weights = parlay::sequence<uint32_t>::from_function(
        ns, [&](size_t i) { return groups[i].size() + 1; });
  } else {
    weights = parlay::sequence<uint32_t>(ns, 1);
  }
  return std::make_pair(std::move(weights), std::move(point_to_sample));
}

template <typename T, typename Range>
auto Wards(const Range &points, uint32_t k, BuildParams BP, double eps = 0.8,
           double delta = 1.0, double over_samples = 20.0,
           bool weighted = true) {
  using PointTy = typename Range::Point;
  parlay::internal::timer t("Wards");
  size_t n = points.size();
  unsigned d = points.get_dims();
  size_t num_samples = std::min((size_t)std::ceil(k * over_samples), n);
  std::cout << "Number of samples: " << num_samples << std::endl;

  if (num_samples == n) {  // Run on all points
    auto dyn_points = dyn::PointRange<PointTy>(points, d);
    auto weights = parlay::sequence<uint32_t>(dyn_points.size(), 1);
    t.next("init points and weights");
    if (eps == 0.0) {  // Exact Wards
      auto [center_ids, assignment] =
          WardsHAC<uint32_t, T>(dyn_points, k, weights);
      auto centers_seq = parlay::delayed_seq<PointTy>(
          center_ids.size(),
          [&](size_t i) { return dyn_points[center_ids[i]]; });
      Range centers(static_cast<const decltype(centers_seq) &>(centers_seq), d);
      t.next("Exact Wards");
      return std::make_pair(std::move(centers), std::move(assignment));
    }
    auto [center_ids, assignment] =
        ApxWardsHAC<uint32_t, T>(dyn_points, k, eps, delta, weights, BP);
    auto centers_seq = parlay::delayed_seq<PointTy>(
        center_ids.size(), [&](size_t i) { return dyn_points[center_ids[i]]; });
    Range centers(static_cast<const decltype(centers_seq) &>(centers_seq), d);
    t.next("Approx Wards");
    return std::make_pair(std::move(centers), std::move(assignment));
  }

  static constexpr uint32_t kDefaultSeed = 42u;
  uint32_t seed = kDefaultSeed;
  size_t d2_rounds = 5;
  auto [ids, p2s] =
      WardsD2Sample<PointTy>(points, num_samples, d2_rounds, seed, BP);
  auto sampled_ids = std::move(ids);
  auto warm_pts = std::move(p2s);

  auto samples = parlay::delayed_seq<PointTy>(
      sampled_ids.size(), [&](size_t i) { return points[sampled_ids[i]]; });
  auto sampled_points = dyn::PointRange<PointTy>(samples, d);
  std::cout << "Sampled points size: " << sampled_points.size() << std::endl;
  t.next("Get sampled points");

  auto [weights, point_to_sample] = compute_weights_anns<PointTy>(
      points, sampled_points, BP, weighted, std::move(warm_pts));
  t.next("Compute weights");

  auto make_centers_and_ids = [&](auto &center_ids, auto &sample_assignment) {
    auto centers_seq = parlay::delayed_seq<PointTy>(
        center_ids.size(),
        [&](size_t i) { return sampled_points[center_ids[i]]; });
    Range centers(static_cast<const decltype(centers_seq) &>(centers_seq), d);
    auto cluster_ids = parlay::sequence<uint32_t>::from_function(
        n, [&](size_t i) { return sample_assignment[point_to_sample[i]]; });
    return std::make_pair(std::move(centers), std::move(cluster_ids));
  };

  if (eps == 0.0) {
    auto [center_ids, sample_assignment] =
        WardsHAC<uint32_t, T>(sampled_points, k, weights);
    t.next("Wards HAC");
    return make_centers_and_ids(center_ids, sample_assignment);
  }
  auto [center_ids, sample_assignment] =
      ApxWardsHAC<uint32_t, T>(sampled_points, k, eps, delta, weights, BP);
  t.next("Wards HAC");
  return make_centers_and_ids(center_ids, sample_assignment);
}
