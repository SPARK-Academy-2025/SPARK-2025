python3 prepare_nnunet.py -i $INPUT_PATH -o $TEMP_PATH/input_nnunet
python3 prepare_mednext.py -i $INPUT_PATH -o $TEMP_PATH/input_mednext
nnUNetv2_predict -i $TEMP_PATH/input_nnunet -o $TEMP_PATH/nnunet -d 501 -c 3d_fullres -f 0 --save_probabilities -device cuda -p nnUNetResEncUNetMPlans
mednextv1_predict -i $TEMP_PATH/input_mednext -o $TEMP_PATH/mednext_raw -tr nnUNetTrainerV2_MedNeXt_M_kernel3 -t 501 -p nnUNetPlansv2.1_trgSp_1x1x1 -m 3d_fullres -z -chk model_best
python3 convert_mednext_npz.py -in_mednext $TEMP_PATH/mednext_raw -in_pkl $TEMP_PATH/nnunet -o $TEMP_PATH/mednext
nnUNetv2_ensemble -i $TEMP_PATH/nnunet $TEMP_PATH/mednext -o $OUTPUT_PATH --use_class_weights
python3 cleanup.py -i $OUTPUT_PATH