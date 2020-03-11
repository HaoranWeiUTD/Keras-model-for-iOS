//
//  OpenCVWrapper.h
//  CameraDemo
//
//  Created by Sehgal, Abhishek on 5/3/18.
//  Copyright © 2018 Sehgal, Abhishek. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

@interface OpenCVWrapper : NSObject
- (UIImage *) preprocessImage: (UIImage *) image image_size: (int) image_size;
+ (CVPixelBufferRef)pixelBufferFromImage:(CGImageRef)image;
@end
