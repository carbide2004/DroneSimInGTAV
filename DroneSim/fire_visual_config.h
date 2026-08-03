#pragma once

namespace FireVisualConfig {

inline char kPtfxAsset[] = "core";
inline char kSmokePtfxEffect[] = "ent_amb_generator_smoke";

constexpr float kMaximumRenderRangeMeters = 500.0f;
constexpr float kSmokeEmitterZOffsetMeters = 0.75f;
constexpr float kSmokeScale = 2.0f;

// This is the visibility proxy for the smoke emitter above. It is
// deliberately narrower than the old generic 8 m x 25 m fire envelope.
constexpr float kSmokeEnvelopeRadiusMeters = 4.0f;
constexpr float kSmokeEnvelopeHeightMeters = 18.0f;

}  // namespace FireVisualConfig
