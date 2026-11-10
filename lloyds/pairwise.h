#pragma once
#include <Eigen/Core>

#include <algorithm>
#include <random>
#include <utility>

#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template <typename Range>
Range copyPoints(const Range &points,
                 const parlay::sequence<uint32_t> &center_ids) {
  using PointTy = typename Range::Point;
  auto centers = parlay::delayed_seq<PointTy>(
      center_ids.size(), [&](size_t i) { return points[center_ids[i]]; });
  unsigned d = points.get_dims();
  return Range(static_cast<const decltype(centers) &>(centers), d);
}

template <typename PointTy, typename Seq, typename Range>
void compute_mean(const Seq &points, Range &centers, uint32_t id, uint32_t d) {
  using StoredT = typename PointTy::T;
  const size_t m = points.size();
  assert(m != 0);

  Eigen::Matrix<double, 1, Eigen::Dynamic> acc64 =
      Eigen::Matrix<double, 1, Eigen::Dynamic>::Zero(1, d);

  for (size_t j = 0; j < m; j++) {
    auto pt = points[j];
    Eigen::Map<const Eigen::Matrix<StoredT, 1, Eigen::Dynamic>> row(&pt[0], 1, d);
    acc64 += row.template cast<double>();
  }

  acc64 /= static_cast<double>(m);

  Eigen::Map<Eigen::Matrix<StoredT, 1, Eigen::Dynamic>> out(&centers[id][0], 1, d);
  out = acc64.template cast<StoredT>();
}

template <typename PointTy, typename DistTy = typename PointTy::distanceType,
          typename Range>
auto compute_centroids_from_clusters(const Range &points,
                                     parlay::sequence<uint32_t> &cluster_ids,
                                     size_t k) {
  size_t n = points.size();
  size_t d = points.get_dims();

  auto min_distance_ids =
      parlay::sequence<std::pair<uint32_t, uint32_t>>::from_function(
          n, [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
  auto grouped = parlay::group_by_index(min_distance_ids, k);
  Range updated_centers(k, d);

  parlay::parallel_for(0, k, [&](size_t i) {
    if (grouped[i].size() > 0) {
      auto subset_points = parlay::delayed_seq<PointTy>(
          grouped[i].size(), [&](size_t j) { return points[grouped[i][j]]; });
      compute_mean<PointTy>(subset_points, updated_centers, i, d);
    }
  });

  auto empty_clusters =
      parlay::filter(parlay::iota<size_t>(k), [&](size_t i) {
        return grouped[i].size() == 0;
      });

  if (empty_clusters.size() > 0) {
    parlay::sequence<parlay::sequence<uint32_t>> pools(k);
    parlay::sequence<size_t> live_sizes(k);
    parlay::parallel_for(0, k, [&](size_t i) {
      live_sizes[i] = grouped[i].size();
      if (grouped[i].size() > 0) {
        pools[i] = parlay::sequence<uint32_t>(grouped[i].begin(),
                                              grouped[i].end());
      }
    });

    parlay::sequence<uint32_t> donors =
        parlay::filter(parlay::iota<uint32_t>(k),
                       [&](uint32_t i) { return live_sizes[i] > 1; });
    std::sort(donors.begin(), donors.end(), [&](uint32_t a, uint32_t b) {
      return live_sizes[a] > live_sizes[b];
    });

    std::mt19937 rng(static_cast<uint32_t>(
        n * 2654435761u + k * 2246822519u +
        empty_clusters.size() * 40503u + donors.size() * 2147483647u));

    size_t reseeded = 0, degenerate = 0;
    size_t dcursor = 0;

    for (size_t ei = 0; ei < empty_clusters.size(); ++ei) {
      const size_t empty_idx = empty_clusters[ei];

      bool found = false;
      uint32_t donor = 0;
      for (size_t t = 0; t < donors.size(); ++t) {
        const size_t c = (dcursor + t) % donors.size();
        if (live_sizes[donors[c]] > 1) {
          donor = donors[c];
          dcursor = (c + 1) % donors.size();
          found = true;
          break;
        }
      }

      if (!found) {
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        const size_t pick = dist(rng);
        auto pt = points[pick];
        for (size_t j = 0; j < d; j++) updated_centers[empty_idx][j] = pt[j];
        degenerate++;
        continue;
      }

      std::uniform_int_distribution<size_t> dist(0, live_sizes[donor] - 1);
      const size_t pick = dist(rng);
      const uint32_t pt_id = pools[donor][pick];
      std::swap(pools[donor][pick], pools[donor][live_sizes[donor] - 1]);
      live_sizes[donor]--;

      auto pt = points[pt_id];
      for (size_t j = 0; j < d; j++) updated_centers[empty_idx][j] = pt[j];
      reseeded++;
    }

    printf("  [warn] %zu empty cluster(s) reseeded from dataset points "
           "(%zu normal, %zu degenerate).\n",
           empty_clusters.size(), reseeded, degenerate);
  }

  return updated_centers;
}

template <typename PointTy, typename DistTy = typename PointTy::distanceType,
          typename Range>
auto compute_cluster_ids_pairwise_blocked(Range &points, Range &centers) {
  const size_t n = points.size();
  const size_t k = centers.size();
  const size_t d = points.get_dims();

  using StoredT = typename PointTy::T;
  using MatRM =
      Eigen::Matrix<DistTy, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using StoredMatRM =
      Eigen::Matrix<StoredT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using StoredStridedMap =
      Eigen::Map<const StoredMatRM, 0, Eigen::OuterStride<Eigen::Dynamic>>;

  const long outer_points = (n > 1)
      ? (&points[1][0] - &points[0][0])
      : static_cast<long>(d);
  const long outer_centers = (k > 1)
      ? (&centers[1][0] - &centers[0][0])
      : static_cast<long>(d);

  const size_t BLOCK_SIZE = std::max(
      size_t(64),
      std::min(size_t(4096), (4UL << 20) / (k * sizeof(DistTy))));

  const StoredStridedMap C_raw(&centers[0][0], k, d,
                               Eigen::OuterStride<Eigen::Dynamic>(outer_centers));
  const MatRM C = C_raw.template cast<DistTy>();

  const Eigen::Matrix<DistTy, 1, Eigen::Dynamic> C_sq_norms =
      C.rowwise().squaredNorm().transpose();

  parlay::sequence<uint32_t> updated_cluster_ids(n);
  const size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  parlay::parallel_for(0, num_blocks, [&](size_t block_idx) {
    const size_t start = block_idx * BLOCK_SIZE;
    const size_t rows  = std::min(BLOCK_SIZE, n - start);

    const StoredStridedMap P_raw(&points[start][0], rows, d,
                                 Eigen::OuterStride<Eigen::Dynamic>(outer_points));
    const MatRM P = P_raw.template cast<DistTy>();

    MatRM D(rows, k);
    D.noalias() = -2 * (P * C.transpose());
    D.rowwise() += C_sq_norms;
    D = (D.array() != D.array())
            .select(std::numeric_limits<DistTy>::infinity(), D.array());

    for (size_t i = 0; i < rows; ++i) {
      Eigen::Index min_idx;
      D.row(i).minCoeff(&min_idx);
      updated_cluster_ids[start + i] = static_cast<uint32_t>(min_idx);
    }
  });

  return updated_cluster_ids;
}

template <typename PointTy, typename DistTy = typename PointTy::distanceType,
          typename Range>
auto lloyds_pairwise(Range &points, Range &centers,
                     parlay::sequence<uint32_t> &cluster_ids) {
  size_t k = centers.size();

  auto updated_centers =
      compute_centroids_from_clusters<PointTy>(points, cluster_ids, k);

  auto updated_cluster_ids =
      compute_cluster_ids_pairwise_blocked<PointTy>(points, updated_centers);

  return std::make_pair(updated_centers, updated_cluster_ids);
}
