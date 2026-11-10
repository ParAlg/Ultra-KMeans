#pragma once
#include "algorithms/utils/graph.h"
#include "algorithms/utils/point_range.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/vamana/index.h"
#include "pairwise.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

using namespace parlayANN;

template <typename PointTy, typename Range>
auto compute_cluster_ids_anns(Range &points, Range &centers, BuildParams &BP,
                              parlay::sequence<uint32_t> cluster_ids = {}) {
  size_t n = points.size();
  size_t k = centers.size();

  const double build_frac = 1.0;
  Graph<uint32_t> G = Graph<uint32_t>(BP.R, k);
  knn_index<Range, Range, uint32_t> I(BP);
  stats<uint32_t> BuildStats(G.size());
  I.set_start();
  auto inserts = parlay::tabulate(k, [](size_t i) { return (uint32_t)i; });
  for (int p = 0; p < BP.num_passes; p++) {
    double a = (p == BP.num_passes - 1) ? BP.alpha : 1.0;
    I.batch_insert(inserts, G, centers, centers, BuildStats, a,
                   /*random_order=*/true, /*base=*/2, build_frac,
                   /*print=*/false);
  }
  parlay::parallel_for(0, G.size(), [&](long i) {
    auto less = [&](uint32_t x, uint32_t y) {
      return centers[i].distance(centers[x]) < centers[i].distance(centers[y]);
    };
    G[i].sort(less);
  });

  double cut = 1.35;
  auto QP = QueryParams(1, BP.L, cut, (long)G.size(), (long)G.max_degree());
  parlay::sequence<uint32_t> updated_cluster_ids(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    parlay::sequence<uint32_t> start_points = {I.get_start()};
    if (!cluster_ids.empty()) {
      start_points.push_back(cluster_ids[i]);
    }
    auto pair_elts = beam_search<PointTy, Range, uint32_t>(
                         points[i], G, centers, start_points, QP)
                         .first;
    auto &visited_elts = pair_elts.second;
    uint32_t min_id = !visited_elts.empty()
                          ? visited_elts[0].first
                          : start_points[0];
    updated_cluster_ids[i] = min_id;
  });
  return updated_cluster_ids;
}

template <typename PointTy, typename DistTy = typename PointTy::distanceType,
          typename Range>
auto lloyds_anns(Range &points, Range &centers,
                 parlay::sequence<uint32_t> &cluster_ids, BuildParams &BP) {
  size_t k = centers.size();
  auto updated_centers =
      compute_centroids_from_clusters<PointTy, DistTy>(points, cluster_ids, k);

  auto updated_cluster_ids = compute_cluster_ids_anns<PointTy>(
      points, updated_centers, BP, cluster_ids);
  return std::make_pair(updated_centers, updated_cluster_ids);
}
