//
//  ObjectDetector.h
//  picodet_demo
//
//  Created by boss on 2025/7/5.
//  Copyright © 2025 reyzhang. All rights reserved.
//  目标检测器

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <PaddleObjectDetector/ObjectDetectorModelInfo.h>
#import <PaddleObjectDetector/DetectedObjectInfo.h>
#import <PaddleObjectDetector/DetectedTimeInfo.h>

NS_ASSUME_NONNULL_BEGIN


@class ObjectDetector;
@class DetectedObjectInfo;
@class DetectedTimeInfo;

//代理方法
@protocol ObjectDetectorDelegate  <NSObject>

@required
- (void)objectDetector:(ObjectDetector *)detector
                result:(NSArray<DetectedObjectInfo *> * )array
                useTime:(DetectedTimeInfo *)time;

@end



@protocol ObjectDetectorDelegate;
@interface ObjectDetector : NSObject

/**
 协议
 */
@property(nonatomic, weak) id<ObjectDetectorDelegate> delegate;

/**
 目标检测模型信息
 */
@property(nonatomic,strong) ObjectDetectorModelInfo *modelInfo;

/**
 指定初始化构造器
 */
- (id)initWithModel:(ObjectDetectorModelInfo *)modelInfo
               delegate:(id<ObjectDetectorDelegate>)delegate;



/**
 目标检测推理
 */
- (void)detectImage:(UIImage *)image;


/**
 获取推理器版本
 */
- (NSString *)getPredicatorVersion;

@end

NS_ASSUME_NONNULL_END
