set OPTION=-b -bc 100
cd ..\\
cd background_removal\\u2net& python u2net.py %OPTION%
cd ..\\..\\depth_estimation\\midas& python midas.py %OPTION%
cd ..\\..\\face_detection\\blazeface& python blazeface.py %OPTION%
cd ..\\..\\face_recognition\\facemesh& python facemesh.py %OPTION%
cd ..\\..\\hand_recognition\\blazehand& python blazehand.py %OPTION%
cd ..\\..\\image_classification\\efficientnet_lite& python efficientnet_lite.py %OPTION%
cd ..\\..\\image_classification\\mobilenetv1& python mobilenetv1.py %OPTION%
cd ..\\..\\image_classification\\mobilenetv2& python mobilenetv2.py %OPTION%
cd ..\\..\\image_classification\\resnet50& python resnet50.py %OPTION%
cd ..\\..\\image_classification\\vgg16& python vgg16.py %OPTION%
cd ..\\..\\image_classification\\squeezenet& python squeezenet.py %OPTION%
cd ..\\..\\image_classification\\efficientnet_lite& python efficientnet_lite.py %OPTION%
cd ..\\..\\image_classification\\googlenet& python googlenet.py %OPTION%
cd ..\\..\\image_segmentation\\deeplabv3plus& python deeplabv3plus.py %OPTION%
cd ..\\..\\image_segmentation\\hrnet_segmentation& python hrnet_segmentation.py %OPTION%
#cd ..\\..\\image_segmentation\\segment-anything-2& python segment-anything-2.py %OPTION%
cd ..\\..\\object_detection\\yolov3-tiny& python yolov3-tiny.py %OPTION%
cd ..\\..\\object_detection\\yolox& python yolox.py %OPTION%
cd ..\\..\\object_detection\\yolox& python yolox.py -m yolox_s %OPTION%
cd ..\\..\\object_detection\\efficientdet_lite& python efficientdet_lite.py %OPTION%
cd ..\\..\\pose_estimation\\pose_resnet& python pose_resnet.py %OPTION%
cd ..\\..\\super_resolution\\srresnet& python srresnet.py %OPTION%
cd ..\\..\\super_resolution\\espcn& python espcn.py %OPTION%
cd ..\\..\\scripts
