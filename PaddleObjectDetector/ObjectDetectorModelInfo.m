//
//  ObjectDetectorModelInfo.m
//  picodet_demo
//
//  Created by boss on 2025/7/5.
//  Copyright © 2025 reyzhang. All rights reserved.
//

#import "ObjectDetectorModelInfo.h"

@implementation ObjectDetectorModelInfo


- (id)init {
    if (self = [super init]) {
        self.modelType = picodet;
        self.input_width = 640;
        self.input_height = 640;
        self.threshold = 0.5;
    }
    return self;
}



@end
