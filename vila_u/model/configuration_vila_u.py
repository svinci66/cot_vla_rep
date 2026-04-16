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
        self.use_hybrid_attention = kwargs.pop("use_hybrid_attention", False)
        self.tune_depth_transformer = kwargs.pop("tune_depth_transformer", False)
