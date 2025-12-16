# nnUNetTrainer_SegMamba.py

import torch
from typing import Union, List, Tuple
from torch._dynamo import OptimizedModule
from nnunetv2.model_segmamba.segmamba import SegMamba
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_SegMamba(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, 
                         device=device)
        
        self.initial_lr = 1e-2 
        self.weight_decay = 1e-5 
        self.model_name = "SegMamba" 
        self.num_epochs = 1000

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True) -> torch.nn.Module:
        segmamba_depths = [2, 2, 2, 2]
        segmamba_feat_size = [48, 96, 192, 384]
        model = SegMamba(
            in_chans=num_input_channels,
            out_chans=num_output_channels,
            depths=segmamba_depths,
            feat_size=segmamba_feat_size,
            do_deep_supervision=enable_deep_supervision 
        )
        return model
    
    def _get_base_model(self):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self._get_base_model()
        mod.do_deep_supervision = enabled

    def save_checkpoint(self, filename: str) -> None:
        """Override to save model without deep supervision weights for inference compatibility."""
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                mod = self._get_base_model()
                original_deep_supervision = getattr(mod, 'do_deep_supervision', True)
                
                try:
                    state_dict = mod.state_dict()
                    
                    # CORRECTED filtering logic for SegMamba:
                    # Remove deep supervision heads ("ds_seg_from_dec...").
                    # Keep everything else, including the main output ("out_main_seg").
                    filtered_state_dict = {
                        k: v for k, v in state_dict.items()
                        if not k.startswith('ds_seg_from_dec')
                    }
                    
                    checkpoint = {
                        'network_weights': filtered_state_dict,
                        'optimizer_state': self.optimizer.state_dict(),
                        'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                        'logging': self.logger.get_checkpoint(),
                        '_best_ema': self._best_ema,
                        'current_epoch': self.current_epoch + 1,
                        'init_args': self.my_init_kwargs,
                        'trainer_name': self.__class__.__name__,
                        'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    }
                    torch.save(checkpoint, filename)
                finally:
                    mod.do_deep_supervision = original_deep_supervision
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """
        Override to load checkpoint, specifically handling deep supervision heads missing
        from checkpoints saved by the custom save_checkpoint method.
        """
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint # Assume it's already a dict

        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            # Handle 'module.' prefix from DataParallel
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # THIS IS THE CRUCIAL PART: Load network weights with strict=False
        print("\nAttempting to load network weights with strict=False (to handle filtered deep supervision heads)...")
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                print("  (DDP, OptimizedModule detected)")
                load_info = self.network.module._orig_mod.load_state_dict(new_state_dict, strict=False)
            else:
                print("  (DDP detected)")
                load_info = self.network.module.load_state_dict(new_state_dict, strict=False)
        else:
            if isinstance(self.network, OptimizedModule):
                print("  (OptimizedModule detected)")
                load_info = self.network._orig_mod.load_state_dict(new_state_dict, strict=False)
            else:
                print("  (Non-DDP/Non-OptimizedModule)")
                load_info = self.network.load_state_dict(new_state_dict, strict=False)

        print(f"  Network loading results:")
        print(f"    Missing keys: {load_info.missing_keys}")
        print(f"    Unexpected keys: {load_info.unexpected_keys}")
        
        # Expectation: load_info.missing_keys should contain 'out_1.weight', 'out_1.bias', 'out_2.weight', etc.
        # This confirms that the deep supervision heads were indeed not loaded, which is correct behavior for your saved checkpoint.
        print("Network weights loaded (potentially partially, as expected for filtered deep supervision heads).\n")

        # Optimizer and grad_scaler are typically loaded strictly
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def perform_actual_validation(self, save_probabilities: bool = False):
        mod = self._get_base_model()
        original_forward = mod.forward
        mod.forward = lambda x: original_forward(x)[0]
        
        original_deep_supervision = getattr(mod, 'do_deep_supervision', True)
        mod.do_deep_supervision = False
        
        try:
            result = super().perform_actual_validation(save_probabilities)
        finally:
            mod.forward = original_forward
            mod.do_deep_supervision = original_deep_supervision
            
        return result

    def validation_step(self, batch: dict) -> dict:
        mod = self._get_base_model()
        original_deep_supervision = getattr(mod, 'do_deep_supervision', True)
        mod.do_deep_supervision = False
        try:
            result = super().validation_step(batch)
        finally:
            mod.do_deep_supervision = original_deep_supervision
        return result

class nnUNetTrainer_SegMamba_50(nnUNetTrainer_SegMamba):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, 
                         device=device)
        
        self.num_epochs = 50

class nnUNetTrainer_SegMamba_500(nnUNetTrainer_SegMamba):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, 
                         device=device)
        
        self.num_epochs = 500


class nnUNetTrainer_SegMamba_1000(nnUNetTrainer_SegMamba):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, 
                         device=device)
        
        self.num_epochs = 1000

'''# nnUNetTrainer_SegMamba.py

import torch
from typing import Union, List, Tuple
from torch._dynamo import OptimizedModule
from nnunetv2.model_segmamba.segmamba import SegMamba
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_SegMamba(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, 
                         device=device)
        
        # Setting optimizers
        self.initial_optimizer_class = torch.optim.SGD
        self.initial_lr = 1e-2 
        self.weight_decay = 1e-5 
        self.model_name = "SegMamba" 

        self.num_epochs = 10

    def build_network_architecture(
            self,
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True) -> torch.nn.Module:
        """
        called by the nnUNetTrainer's initialize() method
        """
        # --- Parameters for SegMamba --- #
        segmamba_depths = [2, 2, 2, 2]
        segmamba_feat_size = [48, 96, 192, 384]

        model = SegMamba(
            in_chans=num_input_channels,
            out_chans=num_output_channels,
            depths=segmamba_depths,
            feat_size=segmamba_feat_size,
            do_deep_supervision=enable_deep_supervision 
        )
        return model
    
    # overriding deepsupervision

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        Handle deep supervision for SegMamba architecture
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
            
        # Unwrap compiled model if needed
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
            
        # Set deep supervision flag directly on SegMamba model
        mod.deep_supervision = enabled

    # optimizer/LR scheduler logic
    # def configure_optimizers(self) -> None:
    #     # if set self.initial_optimizer_class, self.initial_lr, etc., in __init__,
    #     # the superclass's configure_optimizers (called during self.on_train_start()) 
    #     # will likely handle it.
    #     optimizer = torch.optim.AdamW(self.network.parameters(),
    #                                   lr=self.initial_lr, # Use self.initial_lr set in __init__ or from plans
    #                                   weight_decay=self.weight_decay) # Use self.weight_decay
    #     self.optimizer = optimizer
        
    #     # Setup LR scheduler - nnU-Net usually uses PolyLRScheduler
    #     super().configure_optimizers() 
    '''