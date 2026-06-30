from transformers import PretrainedConfig


class VILAUConfig(PretrainedConfig):
    model_type = "vila_u"

    def __init__(
        self,
        llm_cfg=None,
        vision_tower_cfg=None,
        mm_projector_cfg=None,
        architectures=None,
        resume_path=None,
        hidden_size=None,
        mm_hidden_size=None,
        image_aspect_ratio=None,
        num_video_frames=None,
        mm_use_im_start_end=False,
        mm_use_vi_start_end=False,
        mm_use_im_patch_token=True,
        **kwargs
    ):
        super().__init__()

        self.llm_cfg = llm_cfg
        self.vision_tower_cfg = vision_tower_cfg
        self.mm_projector_cfg = mm_projector_cfg
        self.architectures = architectures
        self.resume_path = resume_path
        self.hidden_size = hidden_size
        self.mm_hidden_size = mm_hidden_size
        self.image_aspect_ratio = image_aspect_ratio
        self.num_video_frames = num_video_frames
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_vi_start_end = mm_use_vi_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token

        # ===== Action Prediction =====
        self.action_dim = kwargs.pop("action_dim", 7)
        self.action_chunk_size = kwargs.pop("action_chunk_size", 10)
        self.action_num_bins = kwargs.pop("action_num_bins", 256)
        self.use_action_prediction = kwargs.pop("use_action_prediction", False)
        self.use_discrete_action_prediction = kwargs.pop(
            "use_discrete_action_prediction", False
        )
        self.action_token_ids = kwargs.pop("action_token_ids", None)
        self.action_slot_token_id = kwargs.pop("action_slot_token_id", None)
        self.action_slot_token_ids = kwargs.pop("action_slot_token_ids", None)
        self.action_bin_edges = kwargs.pop("action_bin_edges", None)
        self.use_action_percentile_bins = kwargs.pop("use_action_percentile_bins", False)
        self.action_bin_low_percentile = kwargs.pop("action_bin_low_percentile", 1.0)
        self.action_bin_high_percentile = kwargs.pop("action_bin_high_percentile", 99.0)
        self.use_hybrid_attention = kwargs.pop("use_hybrid_attention", False)
        self.use_visual_cot = kwargs.pop("use_visual_cot", False)
        self.subgoal_min_offset = kwargs.pop("subgoal_min_offset", 1)
        self.subgoal_max_offset = kwargs.pop("subgoal_max_offset", 10)
        self.subgoal_sampling_strategy = kwargs.pop("subgoal_sampling_strategy", "fixed")
        self.use_visual_cot_loss = kwargs.pop("use_visual_cot_loss", False)
        self.visual_loss_weight = kwargs.pop("visual_loss_weight", 1.0)
        self.use_visual_change_weight = kwargs.pop("use_visual_change_weight", False)
        self.visual_change_weight = kwargs.pop("visual_change_weight", 2.0)
        self.action_loss_weight = kwargs.pop("action_loss_weight", 1.0)
        self.tune_depth_transformer = kwargs.pop("tune_depth_transformer", True)
