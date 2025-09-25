//
//  DetectedTimeInfo.h
//  picodet_demo
//
//  Created by boss on 2025/7/5.
//  Copyright © 2025 reyzhang. All rights reserved.
//  目标检测用时

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface DetectedTimeInfo : NSObject


/**
 预处理时间， 单位 ms 毫秒
 */
@property (nonatomic,assign) int pre_process_time;


/**
 推理时间， 单位 ms 毫秒
 */
@property (nonatomic,assign) int predict_time;



/**
 推理结束，拿到结果的时间， 单位 ms 毫秒
 */
@property (nonatomic,assign) int post_process_time;



@end

NS_ASSUME_NONNULL_END
