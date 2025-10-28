import torch
import logging
import copy
import folder_paths
import comfy.utils
import comfy.lora as lora 

logger = logging.getLogger("LLM-SDXL-Adapter")

def model_lora_keys_adapter(adapter_model, prefix="lora_te"):
    """
    Generates a dictionary mapping LoRA keys to the corresponding weight names
    in the LLMToSDXLAdapter model.
    """
    key_map = {}
    # Find linear layers in the adapter, as they are the ones targeted by LoRA
    for name, module in adapter_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # The key in the LoRA file is based on the module name (e.g "wide_blocks.0.linear")
            # It then gets converted to a standard LoRA key format (e.g "lora_te_wide_blocks_0_linear")
            lora_key = f"{prefix}_{name.replace('.', '_')}"
            
            # This LoRA key maps to the original weight tensor in the model
            key_map[lora_key] = f"{name}.weight"
    return key_map


class T5GEMMALORALOADER:
    """
    ComfyUI node that loads a LoRA model and applies it to both a standard
    diffusion model (UNet) and a T5Gemma adapter.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "llm_adapter": ("LLM_ADAPTER",),
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_adapter": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "LLM_ADAPTER")
    FUNCTION = "load_lora"
    CATEGORY = "llm_sdxl"

    def load_lora(self, model, llm_adapter, lora_name, strength_model, strength_adapter):
        # don't do anything if the strength of both is set to 0 
        if strength_model == 0 and strength_adapter == 0:
            return (model, llm_adapter)

        # Load LoRA  
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        # Patch unet
        model_patched = model.clone()
        
        # Generate key mappings for UNet
        key_map_unet = lora.model_lora_keys_unet(model_patched.model)
        
        # Convert LoRA weights to a patch dictionary
        patches_model = lora.load_lora(lora_dict, key_map_unet)
        
        # Apply patches
        model_patched.add_patches(patches_model, strength_model)
        logger.info(f"Applied LoRA '{lora_name}' to UNet with strength {strength_model}")
        
        #  Patch adapter
        adapter_patched = copy.deepcopy(llm_adapter)
        
        
        # Current training code uses lora_te for this module, assume this is the case.
        # Generate  key mappings for our custom adapter (doesn't have to be t5)
        key_map_adapter = model_lora_keys_adapter(adapter_patched, "lora_te")
        
        # Convert LoRA weights to a patch dictionary for the adapter
        patches_adapter = lora.load_lora(lora_dict, key_map_adapter, log_missing=False)

        if not patches_adapter:
            logger.warning(f"No matching keys found in LoRA '{lora_name}' for the T5 Adapter. Returning original adapter.")
        else:
            # Manually apply patches adapter using the calculate_weight function
            adapter_sd = adapter_patched.state_dict()
            original_weights = {} 

            for key, weight in adapter_patched.named_parameters():
                if key in patches_adapter:
                    # Calculate the new weight by applying the patch
                    patch_tuple = (strength_adapter, patches_adapter[key], 1.0, None, None)
                    new_weight = lora.calculate_weight([patch_tuple], weight, key, original_weights=original_weights)

                    adapter_sd[key] = new_weight
            
            adapter_patched.load_state_dict(adapter_sd)
            logger.info(f"Applied LoRA '{lora_name}' to T5 Adapter with strength {strength_adapter}")

        return (model_patched, adapter_patched)


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "T5GEMMALORALOADER": T5GEMMALORALOADER
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T5GEMMALORALOADER": "T5Gemma Load LoRA"
}