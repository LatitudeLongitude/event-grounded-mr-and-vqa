from __future__ import annotations

import pyarrow as pa


STRING_LIST = pa.list_(pa.string())
FLOAT_LIST = pa.list_(pa.float32())


FRAME_ATOMS = pa.schema(
    [
        ("frame_atom_id", pa.string()),
        ("day", pa.string()),
        ("day_num", pa.int16()),
        ("source", pa.string()),
        ("video_hour_stem", pa.string()),
        ("frame_index", pa.int32()),
        ("timestamp_second", pa.float64()),
        ("absolute_second", pa.float64()),
        ("timestamp_hms", pa.string()),
        ("frame_rel_path", pa.string()),
        ("image_path_ref", pa.string()),
        ("tipsv2_embedding_id", pa.string()),
        ("tipsv2_parquet_ref", pa.string()),
        ("tipsv2_row_key", pa.string()),
        ("manifest_ref", pa.string()),
        ("path_exists_at_build", pa.bool_()),
        ("embedding_exists_at_build", pa.bool_()),
        ("is_canonical_frame", pa.bool_()),
        ("canonical_second", pa.float64()),
        ("canonical_exclusion_reason", pa.string()),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


CANONICAL_FRAME_TIMELINE = pa.schema(
    [
        ("frame_atom_id", pa.string()),
        ("day", pa.string()),
        ("day_num", pa.int16()),
        ("source", pa.string()),
        ("canonical_second", pa.float64()),
        ("video_hour_stem", pa.string()),
        ("frame_index", pa.int32()),
        ("is_canonical_frame", pa.bool_()),
        ("canonical_exclusion_reason", pa.string()),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


TEXT_ATOMS = pa.schema(
    [
        ("text_atom_id", pa.string()),
        ("embedding_chunk_id", pa.string()),
        ("day", pa.string()),
        ("day_num", pa.int16()),
        ("source", pa.string()),
        ("start_hour", pa.string()),
        ("end_hour", pa.string()),
        ("hours", STRING_LIST),
        ("raw_start_time", pa.float64()),
        ("raw_end_time", pa.float64()),
        ("time_fragments", pa.string()),
        ("union_time_intervals", pa.string()),
        ("absolute_start_second", pa.float64()),
        ("absolute_end_second", pa.float64()),
        ("aligned_duration_sec", pa.float64()),
        ("time_alignment_status", pa.string()),
        ("qwen3_embedding_id", pa.string()),
        ("qwen3_parquet_ref", pa.string()),
        ("bgem3_embedding_id", pa.string()),
        ("bgem3_parquet_ref", pa.string()),
        ("text_preview", pa.string()),
        ("token_estimate", pa.int32()),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


TEXT_ATOM_FRAGMENTS = pa.schema(
    [
        ("text_atom_id", pa.string()),
        ("source_chunk_id", pa.string()),
        ("day", pa.string()),
        ("source", pa.string()),
        ("hour", pa.string()),
        ("absolute_start_second", pa.float64()),
        ("absolute_end_second", pa.float64()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("timestamp_valid", pa.bool_()),
        ("is_valid_interval", pa.bool_()),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


FEATURE_TABLE = pa.schema(
    [
        ("day", pa.string()),
        ("day_num", pa.int16()),
        ("source", pa.string()),
        ("second", pa.float64()),
        ("anchor_id", pa.string()),
        ("delta_1", pa.float64()),
        ("delta_5", pa.float64()),
        ("delta_15", pa.float64()),
        ("z_delta_1", pa.float64()),
        ("z_delta_5", pa.float64()),
        ("z_delta_15", pa.float64()),
        ("activity_score", pa.float64()),
        ("visual_boundary_score", pa.float64()),
        ("hard_boundary", pa.bool_()),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


TEXT_FEATURE_TABLE = pa.schema(
    [
        ("day", pa.string()),
        ("day_num", pa.int16()),
        ("source", pa.string()),
        ("text_atom_id", pa.string()),
        ("boundary_second", pa.float64()),
        ("dense_shift", pa.float64()),
        ("text_density", pa.float64()),
        ("text_density_change", pa.float64()),
        ("text_semantic_boundary_score", pa.float64()),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


SPARSE_FEATURE_TABLE = pa.schema(
    [
        ("day", pa.string()),
        ("day_num", pa.int16()),
        ("source", pa.string()),
        ("text_atom_id", pa.string()),
        ("boundary_second", pa.float64()),
        ("sparse_overlap", pa.float64()),
        ("sparse_keyword_shift_score", pa.float64()),
        ("top_changed_sparse_vocab_ids", STRING_LIST),
        ("readable_sparse_tokens", STRING_LIST),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


FUSED_BOUNDARIES = pa.schema(
    [
        ("day", pa.string()),
        ("day_num", pa.int16()),
        ("source", pa.string()),
        ("second", pa.float64()),
        ("visual_boundary_score", pa.float64()),
        ("text_semantic_boundary_score", pa.float64()),
        ("sparse_keyword_shift_score", pa.float64()),
        ("temporal_structure_score", pa.float64()),
        ("l1_boundary_score", pa.float64()),
        ("l2_boundary_score", pa.float64()),
        ("l4_boundary_score", pa.float64()),
        ("l5_boundary_score", pa.float64()),
        ("projected_text_atom_ids", STRING_LIST),
        ("projected_sparse_atom_ids", STRING_LIST),
        ("hard_boundary", pa.bool_()),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


EVENT_TABLE = pa.schema(
    [
        ("event_id", pa.string()),
        ("level", pa.string()),
        ("day", pa.string()),
        ("day_num", pa.int16()),
        ("source", pa.string()),
        ("sources", STRING_LIST),
        ("source_role", pa.string()),
        ("source_roles", STRING_LIST),
        ("start_second", pa.float64()),
        ("end_second", pa.float64()),
        ("start_hms", pa.string()),
        ("end_hms", pa.string()),
        ("duration_sec", pa.float64()),
        ("event_kind", pa.string()),
        ("primary_parent_id", pa.string()),
        ("global_event_id", pa.string()),
        ("representative_frame_ids", STRING_LIST),
        ("linked_text_atom_ids", STRING_LIST),
        ("visual_centroid_ref", pa.string()),
        ("text_dense_centroid_ref", pa.string()),
        ("text_sparse_aggregate_ref", pa.string()),
        ("modality_support", STRING_LIST),
        ("confidence", pa.float64()),
        ("confidence_level", pa.string()),
        ("quality_flags", STRING_LIST),
        ("reserved_label_primary", pa.string()),
        ("reserved_label_secondary", pa.string()),
        ("reserved_label_freeform", pa.string()),
        ("reserved_label_confidence", pa.float64()),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


L3_EVENT_TABLE = EVENT_TABLE.append(pa.field("primary_sources", STRING_LIST)).append(pa.field("supporting_sources", STRING_LIST)).append(pa.field("child_source_event_ids", STRING_LIST)).append(pa.field("source_count", pa.int16())).append(pa.field("cross_source_link_ids", STRING_LIST))

L4_L5_EVENT_TABLE = EVENT_TABLE.append(pa.field("child_event_ids", STRING_LIST)).append(pa.field("dominant_sources", STRING_LIST)).append(pa.field("dominant_locations_reserved", STRING_LIST)).append(pa.field("session_order_index", pa.int32())).append(pa.field("contains_parallel_activity", pa.bool_()))


EVENT_EMBEDDINGS = pa.schema(
    [
        ("event_id", pa.string()),
        ("level", pa.string()),
        ("day", pa.string()),
        ("source", pa.string()),
        ("embedding_ref", pa.string()),
        ("embedding", FLOAT_LIST),
        ("source_item_ids", STRING_LIST),
        ("aggregation_method", pa.string()),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


SPARSE_AGGREGATES = pa.schema(
    [
        ("event_id", pa.string()),
        ("level", pa.string()),
        ("day", pa.string()),
        ("source", pa.string()),
        ("embedding_ref", pa.string()),
        ("sparse_vocab_ids", STRING_LIST),
        ("sparse_values", FLOAT_LIST),
        ("aggregation_method", pa.string()),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


LINK_TABLE = pa.schema(
    [
        ("link_id", pa.string()),
        ("event_id", pa.string()),
        ("target_id", pa.string()),
        ("frame_atom_id", pa.string()),
        ("text_atom_id", pa.string()),
        ("embedding_ref", pa.string()),
        ("parent_event_id", pa.string()),
        ("child_event_id", pa.string()),
        ("link_type", pa.string()),
        ("link_role", pa.string()),
        ("score", pa.float64()),
        ("selection_score", pa.float64()),
        ("selection_reason", pa.string()),
        ("selection_rank", pa.int32()),
        ("derived_from_event_id", pa.string()),
        ("derived_from_frame_atom_id", pa.string()),
        ("is_representative", pa.bool_()),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


INVENTORY_STREAMS = pa.schema(
    [
        ("day", pa.string()),
        ("day_num", pa.int16()),
        ("source", pa.string()),
        ("source_role", pa.string()),
        ("modality", pa.string()),
        ("hour", pa.string()),
        ("path_ref", pa.string()),
        ("exists", pa.bool_()),
        ("row_count", pa.int64()),
        ("is_config_seed", pa.bool_()),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


SOURCE_HOUR_COVERAGE = pa.schema(
    [
        ("day", pa.string()),
        ("day_num", pa.int16()),
        ("source", pa.string()),
        ("source_role", pa.string()),
        ("hour", pa.string()),
        ("frames_available", pa.bool_()),
        ("tipsv2_available", pa.bool_()),
        ("text_available", pa.bool_()),
        ("qwen3_available", pa.bool_()),
        ("bgem3_available", pa.bool_()),
        ("missing_modalities", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


CORPUS_INDEX = pa.schema(
    [
        ("event_id", pa.string()),
        ("level", pa.string()),
        ("day", pa.string()),
        ("source", pa.string()),
        ("sources", STRING_LIST),
        ("start_second", pa.float64()),
        ("end_second", pa.float64()),
        ("start_hms", pa.string()),
        ("end_hms", pa.string()),
        ("event_kind", pa.string()),
        ("primary_parent_id", pa.string()),
        ("representative_frame_ids", STRING_LIST),
        ("linked_text_atom_ids", STRING_LIST),
        ("visual_centroid_ref", pa.string()),
        ("text_dense_centroid_ref", pa.string()),
        ("text_sparse_aggregate_ref", pa.string()),
        ("modality_support", STRING_LIST),
        ("confidence", pa.float64()),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


SOURCE_TIMELINE_COVERAGE = pa.schema(
    [
        ("day", pa.string()),
        ("day_num", pa.int16()),
        ("source", pa.string()),
        ("covered_video_sec", pa.float64()),
        ("l2_covered_sec", pa.float64()),
        ("coverage_ratio", pa.float64()),
        ("uncovered_gaps", pa.string()),
        ("quality_flags", STRING_LIST),
        ("created_by", pa.string()),
        ("params_hash", pa.string()),
        ("input_manifest_ref", pa.string()),
    ]
)


SCHEMAS = {
    "frame_atoms": FRAME_ATOMS,
    "canonical_frame_timeline": CANONICAL_FRAME_TIMELINE,
    "text_atoms": TEXT_ATOMS,
    "text_atom_time_fragments": TEXT_ATOM_FRAGMENTS,
    "visual_boundary_scores": FEATURE_TABLE,
    "visual_motion_features": FEATURE_TABLE,
    "text_boundary_scores": TEXT_FEATURE_TABLE,
    "text_topic_features": TEXT_FEATURE_TABLE,
    "sparse_keyword_features": SPARSE_FEATURE_TABLE,
    "fused_boundary_scores": FUSED_BOUNDARIES,
    "l1_micro_events": EVENT_TABLE,
    "l2_source_events": EVENT_TABLE,
    "l3_global_events": L3_EVENT_TABLE,
    "l4_activity_episodes": L4_L5_EVENT_TABLE,
    "l5_sessions": L4_L5_EVENT_TABLE,
    "source_timeline_coverage": SOURCE_TIMELINE_COVERAGE,
    "event_visual_centroids": EVENT_EMBEDDINGS,
    "event_text_dense_centroids": EVENT_EMBEDDINGS,
    "event_text_sparse_aggregates": SPARSE_AGGREGATES,
    "event_frame_links": LINK_TABLE,
    "event_text_links": LINK_TABLE,
    "event_embedding_links": LINK_TABLE,
    "event_parent_child_links": LINK_TABLE,
    "cross_source_links": LINK_TABLE,
    "available_streams": INVENTORY_STREAMS,
    "source_hour_coverage": SOURCE_HOUR_COVERAGE,
    "event_corpus_index": CORPUS_INDEX,
}


def schema_for(name: str) -> pa.Schema:
    return SCHEMAS[name]
