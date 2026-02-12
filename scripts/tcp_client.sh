# ailia MODELS TFLITE TCP CLIENT
export OPTION=""
#export OPTION="--thread_pool_num_threads 1"
python3 ../image_classification/resnet50/resnet50.py -v "127.0.0.1:8006" -s "127.0.0.1:8010" --no_gui ${OPTION} &
python3 ../face_recognition/facemesh/facemesh.py -v "127.0.0.1:8007" -s "127.0.0.1:8011" --no_gui ${OPTION} &
python3 ../object_detection/yolox/yolox.py -v "127.0.0.1:8008" -s "127.0.0.1:8012" --no_gui ${OPTION} &
python3 ../depth_estimation/midas/midas.py -v "127.0.0.1:8009" -s "127.0.0.1:8013" --no_gui ${OPTION} &
wait
