# nnUNetTrainer_MedNext.py

import torch
from typing import Union, List, Tuple
from torch._dynamo import OptimizedModule
from nnunet_mednext import create_mednext_v1
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_MedNext(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, device=device)
        self.model_name = "MedNext" 
        self.num_epochs = 50

    def _get_base_model(self):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    def set_deep_supervision_enabled(self, enabled: bool):
        """Override to set deep_supervision directly on MedNeXt model"""
        mod = self._get_base_model()
        mod.deep_supervision = enabled

    def save_checkpoint(self, filename: str) -> None:
        """Override to save model without deep supervision for inference compatibility"""
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                mod = self.network.module if self.is_ddp else self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod
                
                original_deep_supervision = getattr(mod, 'deep_supervision', True)
                
                try:
                    # Get the full state dictionary
                    state_dict = mod.state_dict()
                    
                    # Create a new dictionary, keeping only the necessary keys.
                    # We filter out the deep supervision keys (out_1, out_2, etc.)
                    # but keep the main output layer keys (out_0).
                    filtered_state_dict = {
                        k: v for k, v in state_dict.items()
                        if not (k.startswith('out_') and not k.startswith('out_0.'))
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
                    # Restore original deep supervision state
                    mod.deep_supervision = original_deep_supervision
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def perform_actual_validation(self, save_probabilities: bool = False):
        mod = self._get_base_model()
        original_forward = mod.forward

        # return only the first element (primary output).
        mod.forward = lambda x: original_forward(x)[0]
        
        # keep the original logic for toggling the deep_supervision flag,
        original_deep_supervision = getattr(mod, 'deep_supervision', True)
        mod.deep_supervision = False
        
        try:
            # use patched forward method, receiving a single tensor.
            result = super().perform_actual_validation(save_probabilities)
        finally:
            # restore the original forward method and the flag
            mod.forward = original_forward
            mod.deep_supervision = original_deep_supervision
            
        return result

    def validation_step(self, batch: dict) -> dict:
        mod = self._get_base_model()
        original_deep_supervision = getattr(mod, 'deep_supervision', True)
        mod.deep_supervision = False
        try:
            result = super().validation_step(batch)
        finally:
            mod.deep_supervision = original_deep_supervision
        return result

class nnUNetTrainer_MedNext_M_3(nnUNetTrainer_MedNext):
    @staticmethod
    def build_network_architecture(*args, **kwargs) -> torch.nn.Module:
        # Extract the arguments we need
        if len(args) >= 5:
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, num_input_channels, num_output_channels = args[:5]
            enable_deep_supervision = args[5] if len(args) > 5 else kwargs.get('enable_deep_supervision', True)
        else:
            # Fallback to kwargs
            num_input_channels = kwargs.get('num_input_channels')
            num_output_channels = kwargs.get('num_output_channels')
            enable_deep_supervision = kwargs.get('enable_deep_supervision', True)
        
        model = create_mednext_v1(
            model_id='M', 
            kernel_size=3, 
            num_classes=num_output_channels,
            num_input_channels=num_input_channels, 
            deep_supervision=enable_deep_supervision 
        )
        return model

class nnUNetTrainer_MedNext_50(nnUNetTrainer_MedNext_M_3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, 
                         device=device)
        
        self.num_epochs = 50

class nnUNetTrainer_MedNext_500(nnUNetTrainer_MedNext_M_3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, 
                         device=device)
        
        self.num_epochs = 500

class nnUNetTrainer_MedNext_1000(nnUNetTrainer_MedNext_M_3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, 
                         device=device)
        
        self.num_epochs = 1000

class nnUNetTrainer_MedNext_M_5(nnUNetTrainer_MedNext):
    @staticmethod
    def build_network_architecture(*args, **kwargs) -> torch.nn.Module:
        # Extract the arguments we need
        if len(args) >= 5:
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, num_input_channels, num_output_channels = args[:5]
            enable_deep_supervision = args[5] if len(args) > 5 else kwargs.get('enable_deep_supervision', True)
        else:
            # Fallback to kwargs
            num_input_channels = kwargs.get('num_input_channels')
            num_output_channels = kwargs.get('num_output_channels')
            enable_deep_supervision = kwargs.get('enable_deep_supervision', True)
        
        model = create_mednext_v1(
            model_id='M', 
            kernel_size=5, 
            num_classes=num_output_channels,
            num_input_channels=num_input_channels, 
            deep_supervision=enable_deep_supervision 
        )
        return model

'''# nnUNetTrainer_MedNext.py

import torch
from typing import Union, List, Tuple
from torch._dynamo import OptimizedModule
from nnunet_mednext import create_mednext_v1
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_MedNext(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, device=device)
        self.model_name = "MedNext" 
        self.num_epochs = 1

    def _get_base_model(self):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    def set_deep_supervision_enabled(self, enabled: bool):
        """Override to set deep_supervision directly on MedNeXt model"""
        mod = self._get_base_model()
        mod.deep_supervision = enabled

    def perform_actual_validation(self, save_probabilities: bool = False):
        mod = self._get_base_model()
        original_forward = mod.forward

        # return only the first element (primary output).
        mod.forward = lambda x: original_forward(x)[0]
        
        # keep the original logic for toggling the deep_supervision flag,
        original_deep_supervision = getattr(mod, 'deep_supervision', True)
        mod.deep_supervision = False
        
        try:
            # use patched forward method, receiving a single tensor.
            result = super().perform_actual_validation(save_probabilities)
        finally:
            # restore the original forward method and the flag
            mod.forward = original_forward
            mod.deep_supervision = original_deep_supervision
            
        return result

    def validation_step(self, batch: dict) -> dict:
        mod = self._get_base_model()
        original_deep_supervision = getattr(mod, 'deep_supervision', True)
        mod.deep_supervision = False
        try:
            result = super().validation_step(batch)
        finally:
            mod.deep_supervision = original_deep_supervision
        return result

class nnUNetTrainer_MedNext_M_3(nnUNetTrainer_MedNext):
    @staticmethod
    def build_network_architecture(architecture_class_name: str, arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        model = create_mednext_v1(
            model_id='M', 
            kernel_size=3, 
            num_classes=num_output_channels,
            num_input_channels=num_input_channels, 
            deep_supervision=enable_deep_supervision 
        )
        return model

class nnUNetTrainer_MedNext_M_5(nnUNetTrainer_MedNext):
    @staticmethod
    def build_network_architecture(architecture_class_name: str, arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        model = create_mednext_v1(
            model_id='M', 
            kernel_size=5, 
            num_classes=num_output_channels,
            num_input_channels=num_input_channels, 
            deep_supervision=enable_deep_supervision 
        )
        return model'''