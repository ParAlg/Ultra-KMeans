package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pairwise",
    hdrs = ["pairwise.h"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      "@parlaylib//parlay:delayed_sequence",
      "@eigen//:eigen",
    ],
)

cc_library(
    name = "anns",
    hdrs = ["anns.h"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      "@parlaylib//parlay:delayed_sequence",
      "@parlayann//algorithms/vamana:index",
      "@parlayann//algorithms/utils:types",
      "@parlayann//algorithms/utils:stats",
      ":pairwise",
    ],
)

cc_binary(
    name = "ultra_kmeans",
    srcs = ["kmeans.cpp"],
    deps = [
      ":pairwise",
      ":anns",
      "//seeding:kmeansplusplus",
      "//seeding:prefixdoubling",
      "//seeding:uniformlyrandom",
      "//seeding:kmeansparallel",
      "//seeding:wards",
      "@parlaylib//parlay:io",
      "@parlaylib//parlay:sequence",
      "@parlayann//algorithms/utils:point_range",
      "@parlayann//algorithms/utils:euclidian_point",
      "@parlayann//algorithms/bench:parse_command_line",
      "@parlayann//algorithms/utils:types",
      "@eigen//:eigen",
    ],
    linkopts = ["-pthread"],
)
