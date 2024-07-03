from torchvision import transforms
from .Utils import *

# ================== Import libs from XMem ==================
add_path(EXTERNAL_ROOT / "XMem")

from model.network import XMem
from inference.inference_core import InferenceCore
from inference.data.mask_mapper import MaskMapper


class XMemWrapper:
    def __init__(
        self,
        single_object=False,
        enable_long_term=True,
        model_type="XMem-s012",
        device="cuda:0",
        debug=False,
    ) -> None:
        self._logger = get_logger(
            log_level="DEBUG" if debug else "INFO", log_name="XMemWrapper"
        )
        self._config = {
            "key_dim": 64,
            "value_dim": 512,
            "hidden_dim": 64,
            "top_k": 30,
            "mem_every": 3,  # r in paper. Increase to improve running speed.
            "deep_update_every": -1,  # Leave -1 normally to synchronize with mem_every
            "single_object": single_object,
            "enable_long_term": enable_long_term,
            "enable_long_term_count_usage": enable_long_term,
            "max_mid_term_frames": 60,  # T_max in paper, decrease to save memory
            "min_mid_term_frames": 3,  # T_min in paper, increase to save memory
            "num_prototypes": 128,  # P in paper
            "max_long_term_elements": 10000,  # LT_max in paper, increase if objects disappear for a long time
        }
        self._device = device

        if model_type not in ["XMem", "XMem-s012", "XMem-no-sensory"]:
            self._logger.warning(
                f"Supported model types are 'XMem', 'XMem-s012' and 'XMem-no-sensory'. Got {model_type}. Will use XMem."
            )
            model_type = "XMem"
        model_path = PROJ_ROOT / f"config/xmem/{model_type}.pth"

        self._network = XMem(self._config, model_path=model_path).to(self._device)
        self._network.eval()

        self._mapper = MaskMapper()

        self._processor = InferenceCore(self._network, config=self._config)

        self._im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def reset(self):
        self._logger.debug("Resetting XMemWrapper...")
        # reset the processor
        self._processor.clear_memory()
        self._processor.all_labels = None
        # reset the mapper
        self._mapper.labels = []
        self._mapper.remappings = {}
        # initialize the first mask
        self._first_mask_loaded = False

    @torch.no_grad()
    def get_mask(self, rgb, mask=None, exhaustive=False):
        """Returns a mask for the given frame.

        Args:
            rgb (np.ndarray): RGB image of the frame.
            mask (np.ndarray, optional): Mask of the frame. Defaults to None.
        """
        with torch.cuda.amp.autocast(enabled=True):
            rgb = self._im_transform(rgb).to(self._device)

            if not self._first_mask_loaded:
                if mask is not None:
                    self._first_mask_loaded = True
                else:
                    return np.zeros(rgb.shape[1:3], dtype=np.uint8)

            if mask is not None:
                mask = mask.astype(np.uint8)
                mask, labels = self._mapper.convert_mask(mask, exhaustive=exhaustive)
                mask = torch.Tensor(mask).to(self._device)
                self._processor.set_all_labels(list(self._mapper.remappings.values()))
            else:
                labels = None

            prob = self._processor.step(rgb, mask, labels)
            torch.cuda.synchronize(device=self._device)

            # probability mask -> index mask
            out_mask = torch.max(prob, dim=0).indices
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
            out_mask = self._mapper.remap_index_mask(out_mask)
            return out_mask
