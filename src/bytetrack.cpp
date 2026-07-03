#include "bytetrack.hpp"

#include <algorithm>
#include <utility>

namespace rkai {
namespace {

float box_iou(const Detection& a, const Detection& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float w = std::max(0.0f, xx2 - xx1);
  const float h = std::max(0.0f, yy2 - yy1);
  const float inter = w * h;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  return inter / (area_a + area_b - inter + 1e-6f);
}

}  // namespace

ByteTracker::ByteTracker(ByteTrackConfig config) : config_(config) {}

std::vector<Detection> ByteTracker::update(const std::vector<Detection>& detections, int64_t frame_id) {
  if (!config_.enabled) {
    return detections;
  }

  current_frame_id_ = frame_id;
  std::vector<int> active_tracks;
  std::vector<int> high_detections;
  std::vector<int> low_detections;

  for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
    if (tracks_[i].state == TrackState::Tracked || tracks_[i].lost_frames <= config_.track_buffer) {
      active_tracks.push_back(i);
    }
  }

  for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
    if (detections[i].score >= config_.high_score_threshold) {
      high_detections.push_back(i);
    } else if (detections[i].score >= config_.low_score_threshold) {
      low_detections.push_back(i);
    }
  }

  std::vector<Detection> output;
  output.reserve(detections.size());

  const auto high_match = match_detections(active_tracks, detections, high_detections, config_.match_iou_threshold);
  update_matched_tracks(detections, high_match.matches, output);

  const auto low_match = match_detections(high_match.unmatched_tracks, detections, low_detections, config_.match_iou_threshold);
  update_matched_tracks(detections, low_match.matches, output);
  mark_lost_tracks(low_match.unmatched_tracks);

  for (const int detection_index : high_match.unmatched_detections) {
    activate_track(detections[detection_index], frame_id, output);
  }

  for (const int detection_index : low_match.unmatched_detections) {
    Detection detection = detections[detection_index];
    detection.track_id = -1;
    output.push_back(detection);
  }

  remove_expired_tracks();
  return output;
}

ByteTracker::MatchResult ByteTracker::match_detections(const std::vector<int>& track_indices,
                                                       const std::vector<Detection>& detections,
                                                       const std::vector<int>& detection_indices,
                                                       float threshold) const {
  struct Candidate {
    int track_index = -1;
    int detection_index = -1;
    float iou = 0.0f;
  };

  std::vector<Candidate> candidates;
  for (const int track_index : track_indices) {
    for (const int detection_index : detection_indices) {
      if (tracks_[track_index].class_id != detections[detection_index].class_id) {
        continue;
      }
      const float score = box_iou(tracks_[track_index].box, detections[detection_index]);
      if (score >= threshold) {
        Candidate candidate;
        candidate.track_index = track_index;
        candidate.detection_index = detection_index;
        candidate.iou = score;
        candidates.push_back(candidate);
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
    return a.iou > b.iou;
  });

  std::vector<int> matched_tracks;
  std::vector<int> matched_detections;
  MatchResult result;
  for (const auto& candidate : candidates) {
    if (std::find(matched_tracks.begin(), matched_tracks.end(), candidate.track_index) != matched_tracks.end()) {
      continue;
    }
    if (std::find(matched_detections.begin(), matched_detections.end(), candidate.detection_index) != matched_detections.end()) {
      continue;
    }
    matched_tracks.push_back(candidate.track_index);
    matched_detections.push_back(candidate.detection_index);
    result.matches.push_back({candidate.track_index, candidate.detection_index});
  }

  for (const int track_index : track_indices) {
    if (std::find(matched_tracks.begin(), matched_tracks.end(), track_index) == matched_tracks.end()) {
      result.unmatched_tracks.push_back(track_index);
    }
  }
  for (const int detection_index : detection_indices) {
    if (std::find(matched_detections.begin(), matched_detections.end(), detection_index) == matched_detections.end()) {
      result.unmatched_detections.push_back(detection_index);
    }
  }
  return result;
}

void ByteTracker::update_matched_tracks(const std::vector<Detection>& detections,
                                        const std::vector<std::pair<int, int>>& matches,
                                        std::vector<Detection>& output) {
  for (const auto& match : matches) {
    auto& track = tracks_[match.first];
    Detection detection = detections[match.second];
    detection.track_id = track.id;

    track.class_id = detection.class_id;
    track.box = detection;
    track.state = TrackState::Tracked;
    track.lost_frames = 0;
    track.last_frame_id = current_frame_id_;

    output.push_back(detection);
  }
}

void ByteTracker::mark_lost_tracks(const std::vector<int>& track_indices) {
  for (const int track_index : track_indices) {
    auto& track = tracks_[track_index];
    track.state = TrackState::Lost;
    ++track.lost_frames;
  }
}

void ByteTracker::remove_expired_tracks() {
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [this](const Track& track) {
                  return track.state == TrackState::Lost && track.lost_frames > config_.track_buffer;
                }),
                tracks_.end());
}

void ByteTracker::activate_track(const Detection& detection, int64_t frame_id, std::vector<Detection>& output) {
  Detection tracked_detection = detection;
  tracked_detection.track_id = next_track_id_;

  Track track;
  track.id = next_track_id_++;
  track.class_id = detection.class_id;
  track.box = tracked_detection;
  track.state = TrackState::Tracked;
  track.lost_frames = 0;
  track.last_frame_id = frame_id;
  tracks_.push_back(track);

  output.push_back(tracked_detection);
}

}  // namespace rkai