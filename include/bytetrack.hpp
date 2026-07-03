#pragma once

#include "pipeline_types.hpp"

#include <cstdint>
#include <vector>

namespace rkai {

struct ByteTrackConfig {
  bool enabled = true;
  float high_score_threshold = 0.50f;
  float low_score_threshold = 0.10f;
  float match_iou_threshold = 0.30f;
  int track_buffer = 30;
};

class ByteTracker {
 public:
  explicit ByteTracker(ByteTrackConfig config = ByteTrackConfig{});

  std::vector<Detection> update(const std::vector<Detection>& detections, int64_t frame_id);

 private:
  enum class TrackState {
    Tracked,
    Lost,
  };

  struct Track {
    int id = -1;
    int class_id = 0;
    Detection box;
    TrackState state = TrackState::Tracked;
    int lost_frames = 0;
    int64_t last_frame_id = -1;
  };

  struct MatchResult {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
  };

  MatchResult match_detections(const std::vector<int>& track_indices,
                               const std::vector<Detection>& detections,
                               const std::vector<int>& detection_indices,
                               float threshold) const;

  void update_matched_tracks(const std::vector<Detection>& detections,
                             const std::vector<std::pair<int, int>>& matches,
                             std::vector<Detection>& output);
  void mark_lost_tracks(const std::vector<int>& track_indices);
  void remove_expired_tracks();
  void activate_track(const Detection& detection, int64_t frame_id, std::vector<Detection>& output);

  ByteTrackConfig config_;
  int next_track_id_ = 1;
  int64_t current_frame_id_ = -1;
  std::vector<Track> tracks_;
};

}  // namespace rkai