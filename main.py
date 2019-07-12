__author__ = 'jmh081701'
import os
import sys
if __name__ == '__main__':
    cmd="& 'E:\Program Files (x86)\Python36\python.exe' " \
        ".\object_detection\model_main.py " \
        "--pipeline_config_path=E:\TempWorkStation\python\pedestrain_detect\pretrained\pipeline.config " \
        "--model_dir=E:\TempWorkStation\python\pedestrain_detect\train " \
        "--num_train_steps=20000 " \
        "--sample_1_of_n_eval_eval_examples=1 " \
        "--alsologtostderr"
    export_graph_cmd=" & 'E:\Program Files (x86)\Python36\python.exe' " \
                     ".\object_detection\export_inference_graph.py " \
                     "--pipeline_config_path=E:\TempWorkStation\python\pedestrain_detect\pretrained\pipeline.config " \
                     "--input_type=image_tensor " \
                     "--trained_checkpoint_prefix=E:\TempWorkStation\python\pedestrain_detect\train\model.ckpt-2393 " \
                     "--output_directory=E:\TempWorkStation\python\pedestrain_detect\train\save_model\\"

    os.system(cmd)
