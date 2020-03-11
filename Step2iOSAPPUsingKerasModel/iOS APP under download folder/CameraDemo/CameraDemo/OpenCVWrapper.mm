//
//  OpenCVWrapper.m
//  CameraDemo
//
//  Created by Sehgal, Abhishek on 5/3/18.
//  Copyright © 2018 Sehgal, Abhishek. All rights reserved.
//

#import "OpenCVWrapper.h"

//OpenCV Imports
#import <opencv2/core.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/imgproc/imgproc.hpp>

using namespace cv;

@implementation OpenCVWrapper

- (UIImage *) preprocessImage:(UIImage *)image image_size: (int) image_size {
    @autoreleasepool{
        Mat inputImage;
        UIImageToMat(image, inputImage);

        if (inputImage.channels() == 1) {
            return image;
        }

        Mat gray;
        Mat resizedImage;
        resize(inputImage, resizedImage, cv::Size(image_size, image_size));
        //cvtColor(resizedImage, resizedImage, COLOR_RGB2BGR);

        //cvtColor(resizedImage, gray, COLOR_BGR2GRAY);
        return MatToUIImage(resizedImage);
    }
}

+ (CVPixelBufferRef)pixelBufferFromImage:(CGImageRef)image {
    @autoreleasepool {
        CGSize frameSize = CGSizeMake(CGImageGetWidth(image), CGImageGetHeight(image));
        CVPixelBufferRef pixelBuffer = NULL;
        CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, frameSize.width, frameSize.height, kCVPixelFormatType_32BGRA, nil, &pixelBuffer);
        if (status != kCVReturnSuccess) {
            return NULL;
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, 0);
        void *data = CVPixelBufferGetBaseAddress(pixelBuffer);
        CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
        CGContextRef context = CGBitmapContextCreate(data, frameSize.width, frameSize.height, 8, CVPixelBufferGetBytesPerRow(pixelBuffer), rgbColorSpace, (CGBitmapInfo) kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
        CGContextDrawImage(context, CGRectMake(0, 0, CGImageGetWidth(image), CGImageGetHeight(image)), image);
        
        CGColorSpaceRelease(rgbColorSpace);
        CGContextRelease(context);
        CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
        
        return pixelBuffer;
        
    }
    
}

@end

