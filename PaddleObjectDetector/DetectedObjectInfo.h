//
//  DetectedObjectInfo.h
//  picodet_demo
//
//  Created by boss on 2025/7/5.
//  Copyright © 2025 reyzhang. All rights reserved.
//  检测后的目标结果信息

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface DetectedObjectInfo : NSObject

/**
 标签分类名称
 */
@property (nonatomic,strong) NSString *class_name;


/**
 置信度评分
 */
@property(nonatomic,assign) CGFloat score;


/**
 目标位置
 */
@property(nonatomic,assign) CGRect rect;
 
 

@end

NS_ASSUME_NONNULL_END
