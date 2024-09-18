CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# ======================================================================================================
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<im_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
VIDEO_PLACEHOLDER = "<video-placeholder>"
# ======================================================================================================

MAX_IMAGE_LENGTH = 16
MAX_VIDEO_LENGTH = 1  # current video datasets only have 1 video?

PAD_LENGTH = 620

# ==================== SGG tokens ===========================================

class SGSpecialTokens:
    VIDEO_FRAME_ID = "#frameid"
    # SG_START = "#sg"
    SG_END = "#sgend"
    SG_SUBJECT = "#subject"
    SG_SUBJECT_ID = "#subid"
    SG_OBJECT = "#object"
    SG_OBJECT_ID = "#objid"
    SG_PREDICATE = "#sgpred"
    SG_BB_START = "#sgbb"
    SG_BB_END = "#sgbbend"
    SG_BB_X1Y1 = "#bbx1y1"
    SG_BB_X2Y2 = "#bbx2y2"
    # SG_BB_X1 = "#sgx1"
    # SG_BB_X2 = "#sgx2"
    # SG_BB_Y1 = "#sgy1"
    # SG_BB_Y2 = "#sgy2"

    @staticmethod
    def get_tokens():
        members = [attr for attr in dir(SGSpecialTokens) if not callable(getattr(SGSpecialTokens, attr)) and not attr.startswith("__")]
        tokens = []
        for mem in members:
            tokens.append(SGSpecialTokens.__getattribute__(SGSpecialTokens,mem))
        return tokens