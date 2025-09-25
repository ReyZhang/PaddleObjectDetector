//
//  ObjectDetectorModelInfo.h
//  picodet_demo
//
//  Created by boss on 2025/7/5.
//  Copyright © 2025 reyzhang. All rights reserved.
//  用于检测的模型信息

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*
 目标检测模型类别
 */
typedef NS_ENUM(NSInteger, ObjectDetectorModelType) {
    picodet = 0,
    ssd_mobilenetv1 = 1,
    yolov3_mobilenet_v3 = 2,
};



@interface ObjectDetectorModelInfo : NSObject
/**
 模型类别
 */
@property (nonatomic, assign) ObjectDetectorModelType  modelType;


/**
 模型所在路径， .nb结尾的文件
 */
@property(nonatomic,strong) NSString *modelPath;


/**
 标签所在路径， .txt结尾的文件
 */
@property(nonatomic,strong) NSString *labelsPath;


/**
 训练模型时指定的width, 可通过导出的推理模型中的infer_cfg.yml 中查看
 */
@property(nonatomic,assign) int input_width;


/**
 训练模型时指定的height, 可通过导出的推理模型中的infer_cfg.yml 中查看
 */
@property(nonatomic,assign) int input_height;



/**
 训练模型时指定的阈值（置信度（score）阈值）, 可通过导出的推理模型中的infer_cfg.yml 中查看
 */
@property(nonatomic,assign) CGFloat threshold;

@end

NS_ASSUME_NONNULL_END
