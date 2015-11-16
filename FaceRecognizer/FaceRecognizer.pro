TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

INCLUDEPATH += -I/usr/include/opencv

LIBS += -L/lib64 -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab -ltbb -lGL -lGLU -lrt -lpthread -lm -ldl
